import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class EncoderModule(nn.Module):
    """文本编码器模块"""
    def __init__(self, config):
        super(EncoderModule, self).__init__()
        self.config = config

        # 从本地文件夹加载预训练模型
        pretrained_path = f"{config.pretrained_model_path}"
        self.encoder = AutoModel.from_pretrained(pretrained_path)

        # 特征增强层
        if config.use_feature_enhancer:
            self.feature_enhancer = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(config.dropout_rate),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )

    def forward(self, input_ids, attention_mask):
        """
        编码输入文本

        input_ids: [batch_size, seq_len] 输入token IDs
        attention_mask: [batch_size, seq_len] 注意力掩码

        返回: [batch_size, seq_len, hidden_size] 隐藏状态
        """
        # 编码文本
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # 使用特征增强层
        if self.config.use_feature_enhancer:
            hidden_states = self.feature_enhancer(hidden_states)

        return hidden_states
