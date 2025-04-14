import os
import torch
from dataclasses import dataclass


@dataclass
class Config:
    # 基础配置
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 路径配置
    dataset_path: str = "datasets"  # 数据集路径
    pretrained_model_path: str = "pretrained_models/bert-base-chinese"  # 预训练模型路径
    output_dir: str = "outputs"  # 输出路径

    # 模型配置
    hidden_size: int = 768  # 隐藏层大小
    dropout_rate: float = 0.1  # Dropout比例
    use_feature_enhancer: bool = True  # 是否使用特征增强层
    use_aag: bool = True  # 是否使用论元关联图
    use_gnd: bool = True  # 是否使用全局归一化解码

    # 训练配置
    batch_size: int = 16  # 训练批次大小
    eval_batch_size: int = 32  # 评估批次大小
    epochs: int = 10  # 训练轮数
    learning_rate: float = 2e-5  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    warmup_ratio: float = 0.1  # 预热比例
    gradient_accumulation_steps: int = 1  # 梯度累积步数

    # 文本处理配置
    max_length: int = 128  # 最大序列长度

    def __post_init__(self):
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)


def get_config():
    return Config()
