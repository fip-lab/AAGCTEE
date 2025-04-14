import os
import sys
import json
import torch
import argparse
from config.config import get_config
from data.processor import EventDataProcessor
from models.aagctee import AAGCTEE
from test import load_model, predict_single


def parse_args():
    parser = argparse.ArgumentParser(description="基于论元关联图的级联类型事件抽取模型预测")
    parser.add_argument("--text", type=str, help="要预测的文本")
    parser.add_argument("--file", type=str, help="包含要预测的文本的文件")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径，默认使用最佳模型")
    parser.add_argument("--output", type=str, default="predictions.json", help="输出文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()

    # 确保有输入文本
    if args.text is None and args.file is None:
        print("错误：必须提供文本(--text)或文件(--file)作为输入")
        sys.exit(1)

    # 准备数据处理器
    processor = EventDataProcessor(config)

    # 加载模型
    model_path = args.model_path or os.path.join(config.output_dir, "best_model", "model.pt")
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在 {model_path}")
        sys.exit(1)

    print(f"加载模型: {model_path}")
    model = load_model(model_path, config, processor)
    model.to(config.device)

    # 准备输入文本
    texts = []
    if args.text:
        texts.append(args.text)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    # 进行预测
    print(f"开始预测 {len(texts)} 个文本样本...")
    results = []

    for idx, text in enumerate(texts):
        if idx % 10 == 0 and idx > 0:
            print(f"已完成 {idx}/{len(texts)} 个样本")

        events = predict_single(text, model, processor, config)
        results.append({
            "text": text,
            "events": events
        })

    # 保存预测结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"预测完成，结果已保存到 {args.output}")

    # 打印第一个预测结果作为示例
    if results:
        print("\n预测示例:")
        print(f"文本: {results[0]['text']}")
        if results[0]['events']:
            for event in results[0]['events']:
                print(f"事件类型: {event['event_type']}")
                print(f"触发词: {event['trigger']['text']} ({event['trigger']['start']}:{event['trigger']['end']})")
                print("论元:")
                for arg in event['arguments']:
                    print(f"  {arg['role']}: {arg['text']} ({arg['start']}:{arg['end']})")
        else:
            print("未检测到事件")


if __name__ == "__main__":
    main()
