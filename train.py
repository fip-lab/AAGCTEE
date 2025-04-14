import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import logging
import json
from datetime import datetime

from config.config import get_config
from data.processor import EventDataProcessor
from data.dataset import EventExtractionDataset, collate_fn, get_data_loaders
from models.aagctee import AAGCTEE
from utils.metrics import compute_metrics


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """设置日志"""
    log_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def convert_predictions_to_events(trigger_probs, argument_probs, texts, offset_mappings, processor, config):
    """将模型预测转换为事件抽取结果"""
    batch_size = trigger_probs.size(0)
    predictions = []

    for i in range(batch_size):
        sample_events = []

        # 阈值判断，提取触发词
        for event_type_idx in range(trigger_probs.size(3)):
            for start_idx in range(trigger_probs.size(1)):
                for end_idx in range(start_idx, trigger_probs.size(2)):
                    if trigger_probs[i, start_idx, end_idx, event_type_idx] > 0.5:
                        event_type = processor.event_types[event_type_idx]

                        # 通过offset_mapping从token位置转换到原始文本位置
                        char_start = offset_mappings[i][start_idx][0].item()
                        char_end = offset_mappings[i][end_idx][1].item()

                        if char_start == 0 and char_end == 0:
                            continue  # 跳过特殊token

                        trigger_text = texts[i][char_start:char_end]

                        # 找出对应事件类型的论元
                        arguments = []
                        for role_type_idx in range(argument_probs.size(2)):
                            for arg_start in range(argument_probs.size(3)):
                                for arg_end in range(arg_start, argument_probs.size(4)):
                                    if argument_probs[i, event_type_idx, role_type_idx, arg_start, arg_end] > 0.5:
                                        role_type = processor.role_types[role_type_idx]

                                        # 转换论元位置
                                        arg_char_start = offset_mappings[i][arg_start][0].item()
                                        arg_char_end = offset_mappings[i][arg_end][1].item()

                                        if arg_char_start == 0 and arg_char_end == 0:
                                            continue  # 跳过特殊token

                                        arg_text = texts[i][arg_char_start:arg_char_end]

                                        arguments.append({
                                            "role": role_type,
                                            "start": arg_char_start,
                                            "end": arg_char_end,
                                            "text": arg_text
                                        })

                        # 构建事件
                        event = {
                            "event_type": event_type,
                            "trigger": {
                                "start": char_start,
                                "end": char_end,
                                "text": trigger_text
                            },
                            "arguments": arguments
                        }

                        sample_events.append(event)

        # 整合当前样本的预测结果
        sample_pred = {
            "text": texts[i],
            "triggers": [(e["event_type"], e["trigger"]["start"], e["trigger"]["end"]) for e in sample_events],
            "arguments": [(e["event_type"], a["start"], a["end"], a["role"]) for e in sample_events for a in
                          e["arguments"]]
        }

        predictions.append(sample_pred)

    return predictions


def convert_labels_to_events(trigger_labels, argument_labels, texts, offset_mappings, processor):
    """将标签转换为事件抽取结果"""
    batch_size = trigger_labels.size(0)
    labels = []

    for i in range(batch_size):
        sample_events = []

        # 从标签中提取触发词
        for event_type_idx in range(trigger_labels.size(1)):
            for start_idx in range(trigger_labels.size(2)):
                for end_idx in range(start_idx, trigger_labels.size(3)):
                    if trigger_labels[i, event_type_idx, start_idx, end_idx] > 0.5:
                        event_type = processor.event_types[event_type_idx]

                        # 转换token位置到原始文本位置
                        char_start = offset_mappings[i][start_idx][0].item()
                        char_end = offset_mappings[i][end_idx][1].item()

                        if char_start == 0 and char_end == 0:
                            continue  # 跳过特殊token

                        # 找出对应事件类型的论元
                        arguments = []
                        for role_type_idx in range(argument_labels.size(2)):
                            for arg_start in range(argument_labels.size(3)):
                                for arg_end in range(arg_start, argument_labels.size(4)):
                                    if argument_labels[i, event_type_idx, role_type_idx, arg_start, arg_end] > 0.5:
                                        role_type = processor.role_types[role_type_idx]

                                        # 转换论元位置
                                        arg_char_start = offset_mappings[i][arg_start][0].item()
                                        arg_char_end = offset_mappings[i][arg_end][1].item()

                                        if arg_char_start == 0 and arg_char_end == 0:
                                            continue

                                        arguments.append({
                                            "role": role_type,
                                            "start": arg_char_start,
                                            "end": arg_char_end
                                        })

                        # 构建事件
                        event = {
                            "event_type": event_type,
                            "trigger": {
                                "start": char_start,
                                "end": char_end
                            },
                            "arguments": arguments
                        }

                        sample_events.append(event)

        # 整合当前样本的标签结果
        sample_label = {
            "text": texts[i],
            "triggers": [(e["event_type"], e["trigger"]["start"], e["trigger"]["end"]) for e in sample_events],
            "arguments": [(e["event_type"], a["start"], a["end"], a["role"]) for e in sample_events for a in
                          e["arguments"]]
        }

        labels.append(sample_label)

    return labels


def train_model(config):
    """训练AAGCTEE模型"""
    # 设置随机种子和日志
    set_seed(config.seed)
    logger = setup_logging(config)
    logger.info(f"使用设备: {config.device}")
    logger.info(f"模型配置: {config.__dict__}")

    # 准备数据
    logger.info("加载数据...")
    processor = EventDataProcessor(config)
    train_loader, dev_loader, test_loader = get_data_loaders(config, processor)
    logger.info(
        f"加载完成，训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(dev_loader.dataset)}, 测试样本数: {len(test_loader.dataset)}")

    # 加载模型
    logger.info("初始化模型...")
    model = AAGCTEE(
        config,
        num_event_types=len(processor.event_types),
        num_role_types=len(processor.role_types)
    )
    model.to(config.device)
    logger.info(f"事件类型数量: {len(processor.event_types)}, 角色类型数量: {len(processor.role_types)}")

    # 优化器和学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps
    )

    # 训练循环
    logger.info("开始训练...")
    best_f1 = 0
    best_metrics = None

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # 训练
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            # 移动数据到设备
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            trigger_labels = batch["trigger_labels"].to(config.device)
            argument_labels = batch["argument_labels"].to(config.device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                trigger_labels=trigger_labels,
                argument_labels=argument_labels
            )

            loss = outputs["loss"]

            # 反向传播
            loss.backward()

            # 梯度累积
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item()

            # 打印训练进度
            if (step + 1) % 100 == 0:
                logger.info(f"  Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"  Average training loss: {avg_train_loss:.4f}")

        # 验证
        logger.info("开始验证...")
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dev_loader:
                # 移动数据到设备
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)

                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # 处理预测结果
                trigger_probs = outputs["trigger_probs"]
                argument_probs = outputs["argument_probs"]

                # 将预测结果转换为事件抽取格式
                predictions = convert_predictions_to_events(
                    trigger_probs, argument_probs, batch["texts"],
                    batch["offset_mappings"], processor, config
                )
                all_predictions.extend(predictions)

                # 收集真实标签
                labels = convert_labels_to_events(
                    batch["trigger_labels"], batch["argument_labels"],
                    batch["texts"], batch["offset_mappings"], processor
                )
                all_labels.extend(labels)

        # 计算评估指标
        metrics = compute_metrics(
            all_predictions, all_labels,
            processor.event_types, processor.role_types
        )

        logger.info("验证结果:")
        for task in ['TI', 'TC', 'AI', 'AC']:
            logger.info(
                f"  {task} - P: {metrics[f'{task}_precision']:.2f}%, R: {metrics[f'{task}_recall']:.2f}%, F1: {metrics[f'{task}_f1']:.2f}%")
        logger.info(f"  Average F1: {metrics['Avg_F1']:.2f}%")

        # 保存最佳模型
        if metrics['Avg_F1'] > best_f1:
            best_f1 = metrics['Avg_F1']
            best_metrics = metrics

            model_dir = os.path.join(config.output_dir, "best_model")
            os.makedirs(model_dir, exist_ok=True)

            # 保存模型
            model_path = os.path.join(model_dir, "model.pt")
            torch.save(model.state_dict(), model_path)

            # 保存配置
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config.__dict__, f, indent=2)

            logger.info(f"保存最佳模型到 {model_path}, F1: {best_f1:.2f}%")

    # 训练结束，输出最佳结果
    logger.info("训练结束，最佳验证结果:")
    for task in ['TI', 'TC', 'AI', 'AC']:
        logger.info(
            f"  {task} - P: {best_metrics[f'{task}_precision']:.2f}%, R: {best_metrics[f'{task}_recall']:.2f}%, F1: {best_metrics[f'{task}_f1']:.2f}%")
    logger.info(f"  Average F1: {best_metrics['Avg_F1']:.2f}%")

    return model, best_metrics


if __name__ == "__main__":
    config = get_config()
    train_model(config)
