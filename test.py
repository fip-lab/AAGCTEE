import os
import json
import torch
import logging
from datetime import datetime

from config.config import get_config
from data.processor import EventDataProcessor
from data.dataset import get_data_loaders
from models.aagctee import AAGCTEE
from utils.metrics import compute_metrics
from train import convert_predictions_to_events, convert_labels_to_events, set_seed


def setup_logging(config):
    """设置日志"""
    log_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_{timestamp}.log")

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


def load_model(model_path, config, processor):
    """加载训练好的模型"""
    model = AAGCTEE(
        config,
        num_event_types=len(processor.event_types),
        num_role_types=len(processor.role_types)
    )
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    return model


def test_model(config, model_path=None):
    """测试模型性能"""
    # 设置随机种子和日志
    set_seed(config.seed)
    logger = setup_logging(config)
    logger.info(f"使用设备: {config.device}")

    # 准备数据
    logger.info("加载数据...")
    processor = EventDataProcessor(config)
    _, _, test_loader = get_data_loaders(config, processor)
    logger.info(f"测试样本数: {len(test_loader.dataset)}")

    # 加载模型
    if model_path is None:
        model_path = os.path.join(config.output_dir, "best_model", "model.pt")

    logger.info(f"加载模型: {model_path}")
    model = load_model(model_path, config, processor)
    model.eval()

    # 测试
    logger.info("开始测试...")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
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

    # 输出测试结果
    logger.info("测试结果:")
    for task in ['TI', 'TC', 'AI', 'AC']:
        logger.info(
            f"  {task} - P: {metrics[f'{task}_precision']:.2f}%, R: {metrics[f'{task}_recall']:.2f}%, F1: {metrics[f'{task}_f1']:.2f}%")
    logger.info(f"  Average F1: {metrics['Avg_F1']:.2f}%")

    # 保存测试结果
    results_dir = os.path.join(config.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"test_results_{timestamp}.json")

    with open(results_path, "w") as f:
        json.dump({
            "metrics": {k: float(v) for k, v in metrics.items()},
            "config": config.__dict__,
        }, f, indent=2)

    logger.info(f"测试结果已保存到 {results_path}")

    # 保存预测结果
    predictions_path = os.path.join(results_dir, f"predictions_{timestamp}.json")

    formatted_predictions = []
    for pred, text in zip(all_predictions, [label["text"] for label in all_labels]):
        formatted_events = []
        for event_type, start, end in pred["triggers"]:
            event = {
                "event_type": event_type,
                "trigger": {
                    "text": text[start:end],
                    "start": start,
                    "end": end
                },
                "arguments": []
            }

            # 找出该触发词的所有论元
            for e_type, arg_start, arg_end, role in pred["arguments"]:
                if e_type == event_type:
                    event["arguments"].append({
                        "role": role,
                        "text": text[arg_start:arg_end],
                        "start": arg_start,
                        "end": arg_end
                    })

            formatted_events.append(event)

        formatted_predictions.append({
            "text": text,
            "events": formatted_events
        })

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(formatted_predictions, f, ensure_ascii=False, indent=2)

    logger.info(f"预测结果已保存到 {predictions_path}")

    return metrics


def predict_single(text, model, processor, config):
    """预测单个文本样本"""
    # 编码文本
    encoding = processor.tokenizer(
        text,
        max_length=config.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    input_ids = encoding.input_ids.to(config.device)
    attention_mask = encoding.attention_mask.to(config.device)
    offset_mapping = encoding.offset_mapping

    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # 处理预测结果
    trigger_probs = outputs["trigger_probs"]
    argument_probs = outputs["argument_probs"]

    # 将预测结果转换为事件抽取格式
    predictions = convert_predictions_to_events(
        trigger_probs, argument_probs, [text],
        [offset_mapping.squeeze(0)], processor, config
    )

    # 格式化输出
    formatted_events = []
    for event_type, start, end in predictions[0]["triggers"]:
        event = {
            "event_type": event_type,
            "trigger": {
                "text": text[start:end],
                "start": start,
                "end": end
            },
            "arguments": []
        }

        # 找出该触发词的所有论元
        for e_type, arg_start, arg_end, role in predictions[0]["arguments"]:
            if e_type == event_type:
                event["arguments"].append({
                    "role": role,
                    "text": text[arg_start:arg_end],
                    "start": arg_start,
                    "end": arg_end
                })

        formatted_events.append(event)

    return formatted_events


if __name__ == "__main__":
    config = get_config()
    test_model(config)
