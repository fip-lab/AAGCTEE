import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer


class EventDataProcessor:
    """事件抽取数据处理器"""

    def __init__(self, config):
        self.config = config

        # 从本地文件夹加载tokenizer
        pretrained_path = f"{config.pretrained_model_path}"
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

        # 加载数据集元信息
        self.event_types = self.load_event_types()
        self.role_types = self.load_role_types()
        self.event_type_to_idx = {event_type: idx for idx, event_type in enumerate(self.event_types)}
        self.role_type_to_idx = {role_type: idx for idx, role_type in enumerate(self.role_types)}

    def load_event_types(self):
        """加载事件类型"""
        with open(os.path.join(self.config.dataset_path, "event_types.json"), "r", encoding="utf-8") as f:
            event_types = json.load(f)
        return event_types

    def load_role_types(self):
        """加载角色类型"""
        with open(os.path.join(self.config.dataset_path, "role_types.json"), "r", encoding="utf-8") as f:
            role_types = json.load(f)
        return role_types

    def load_data(self, split):
        """
        加载数据集

        split: 数据集划分 ("train", "dev", "test")
        """
        file_path = os.path.join(self.config.dataset_path, f"{split}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def process_sample(self, sample):
        """
        处理单个数据样本

        sample: 原始数据样本
        """
        text = sample["text"]
        events = sample.get("events", [])

        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        offset_mapping = encoding.offset_mapping.squeeze(0)

        # 创建触发词标签张量 [num_event_types, seq_len, seq_len]
        seq_len = input_ids.size(0)
        trigger_labels = torch.zeros(len(self.event_types), seq_len, seq_len)

        # 创建论元标签张量 [num_event_types, num_role_types, seq_len, seq_len]
        argument_labels = torch.zeros(len(self.event_types), len(self.role_types), seq_len, seq_len)

        # 填充触发词和论元标签
        for event in events:
            event_type = event["event_type"]
            event_type_idx = self.event_type_to_idx[event_type]

            # 处理触发词
            trigger = event["trigger"]
            trigger_start, trigger_end = self.get_token_indices(trigger["start"], trigger["end"], offset_mapping)

            if trigger_start is not None and trigger_end is not None:
                trigger_labels[event_type_idx, trigger_start, trigger_end] = 1.0

            # 处理论元
            for argument in event.get("arguments", []):
                role_type = argument["role"]
                role_type_idx = self.role_type_to_idx[role_type]
                arg_start, arg_end = self.get_token_indices(argument["start"], argument["end"], offset_mapping)

                if arg_start is not None and arg_end is not None:
                    argument_labels[event_type_idx, role_type_idx, arg_start, arg_end] = 1.0

        return {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "trigger_labels": trigger_labels,
            "argument_labels": argument_labels
        }

    def get_token_indices(self, char_start, char_end, offset_mapping):
        """
        获取字符级别的起止位置对应的token级别的起止位置
        """
        token_start = None
        token_end = None

        for token_idx, (token_char_start, token_char_end) in enumerate(offset_mapping):
            # 跳过特殊token
            if token_char_start == token_char_end == 0:
                continue

            # 找到包含起始位置的token
            if token_start is None and token_char_start <= char_start < token_char_end:
                token_start = token_idx

            # 找到包含结束位置的token
            if token_char_start < char_end <= token_char_end:
                token_end = token_idx
                break

        return token_start, token_end
