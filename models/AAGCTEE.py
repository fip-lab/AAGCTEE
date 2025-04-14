import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import EncoderModule
from models.trigger_module import TriggerRecognitionModule
from models.argument_module import ArgumentExtractionModule


class AAGCTEE(nn.Module):
    """基于论元关联图的级联类型事件抽取模型"""

    def __init__(self, config, num_event_types, num_role_types):
        super(AAGCTEE, self).__init__()
        self.config = config
        self.num_event_types = num_event_types
        self.num_role_types = num_role_types

        # 编码器模块
        self.encoder = EncoderModule(config)

        # 触发词识别分类模块
        self.trigger_module = TriggerRecognitionModule(config)

        # 论元识别分类模块
        self.argument_module = ArgumentExtractionModule(config, num_event_types, num_role_types)

        # 学习阈值参数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, trigger_labels=None, argument_labels=None):
        """
        模型前向传播

        input_ids: [batch_size, seq_len] 输入token IDs
        attention_mask: [batch_size, seq_len] 注意力掩码
        trigger_labels: [batch_size, num_event_types, seq_len, seq_len] 触发词标签
        argument_labels: [batch_size, num_event_types, num_role_types, seq_len, seq_len] 论元标签

        返回: 损失值和预测结果
        """
        # 编码输入文本
        hidden_states = self.encoder(input_ids, attention_mask)

        # 触发词识别分类
        trigger_probs = self.trigger_module(hidden_states)

        # 从触发词概率中提取触发词span
        if trigger_labels is not None and self.training:
            # 训练时使用真实的触发词span
            trigger_spans, event_types = self._extract_trigger_spans_from_labels(trigger_labels)
        else:
            # 预测时从触发词预测结果中提取span
            trigger_spans, event_types = self._extract_trigger_spans_from_probs(trigger_probs)

        # 论元识别分类
        argument_probs = self.argument_module(hidden_states, trigger_spans, event_types)

        # 计算损失
        loss = None
        if trigger_labels is not None and argument_labels is not None and self.training:
            # 触发词分类损失
            trigger_loss = self._compute_trigger_loss(trigger_probs, trigger_labels)

            # 论元分类损失
            argument_loss = self._compute_argument_loss(argument_probs, argument_labels)

            # 阈值正则化损失
            reg_loss = torch.abs(self.gamma)

            # 总损失
            loss = trigger_loss + argument_loss + 0.01 * reg_loss

        return {
            'loss': loss,
            'trigger_probs': trigger_probs,
            'argument_probs': argument_probs
        }

    def _extract_trigger_spans_from_labels(self, trigger_labels):
        """从触发词标签中提取触发词span和事件类型"""
        batch_size = trigger_labels.size(0)

        trigger_spans = []
        event_types = []

        for b in range(batch_size):
            batch_spans = []
            batch_types = []

            for event_type_idx in range(self.num_event_types):
                for i in range(trigger_labels.size(2)):
                    for j in range(i, trigger_labels.size(3)):
                        if trigger_labels[b, event_type_idx, i, j] > 0.5:
                            batch_spans.append((i, j))
                            batch_types.append(event_type_idx)

            trigger_spans.append(batch_spans)
            event_types.append(batch_types)

        return trigger_spans, event_types

    def _extract_trigger_spans_from_probs(self, trigger_probs):
        """从触发词预测概率中提取触发词span和事件类型"""
        batch_size = trigger_probs.size(0)

        trigger_spans = []
        event_types = []

        for b in range(batch_size):
            batch_spans = []
            batch_types = []

            for event_type_idx in range(self.num_event_types):
                for i in range(trigger_probs.size(1)):
                    for j in range(i, trigger_probs.size(2)):
                        if trigger_probs[b, i, j, event_type_idx] > self.gamma:
                            batch_spans.append((i, j))
                            batch_types.append(event_type_idx)

            trigger_spans.append(batch_spans)
            event_types.append(batch_types)

        return trigger_spans, event_types

    def _compute_trigger_loss(self, trigger_probs, trigger_labels):
        """计算触发词分类损失"""
        # 二元交叉熵损失
        loss = F.binary_cross_entropy(trigger_probs, trigger_labels.float())
        return loss

    def _compute_argument_loss(self, argument_probs, argument_labels):
        """计算论元分类损失"""
        # 获取正样本和负样本
        positives = (argument_labels == 1).float()
        negatives = (argument_labels == 0).float()

        # 计算损失
        pos_loss = -torch.log(argument_probs + 1e-10) * positives
        neg_loss = -torch.log(1 - argument_probs + 1e-10) * negatives

        # 合并损失
        loss = (pos_loss.sum() + neg_loss.sum()) / (positives.sum() + negatives.sum() + 1e-10)

        return loss
