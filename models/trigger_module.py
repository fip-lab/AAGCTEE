import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConditionalLayerNorm(nn.Module):
    """条件层归一化"""

    def __init__(self, hidden_size, eps=1e-12):
        super(ConditionalLayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, cond_x):
        """x: [batch, seq_len, hidden_size], cond_x: [batch, seq_len, hidden_size]"""
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        std = (var + self.eps).sqrt()

        x = (x - mean) / std

        return x * self.weight + self.bias


class DilatedConvolution(nn.Module):
    """空洞卷积层"""

    def __init__(self, hidden_size, dilation=1, kernel_size=3):
        super(DilatedConvolution, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )

    def forward(self, x):
        """x: [batch_size, seq_len, seq_len, hidden_size]"""
        batch_size, seq_len, _, hidden_size = x.size()

        # 调整维度顺序以适应卷积操作 [batch_size, hidden_size, seq_len, seq_len]
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)

        # 恢复原始维度顺序 [batch_size, seq_len, seq_len, hidden_size]
        x = x.permute(0, 2, 3, 1)

        return x


class CascadeTypePredictor(nn.Module):
    """级联类型预测器"""

    def __init__(self, hidden_size, num_event_types):
        super(CascadeTypePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_event_types = num_event_types

        # 双仿射分类器参数
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))

        # MLP参数
        self.mlp1 = nn.Linear(hidden_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, num_event_types)

        # 第二个预测器
        self.mlp1_2 = nn.Linear(hidden_size, hidden_size)
        self.mlp2_2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3_2 = nn.Linear(hidden_size, num_event_types)
        self.W_2 = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_2 = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, grid_repr):
        """
        grid_repr: [batch_size, seq_len, seq_len, hidden_size]
        """
        batch_size, seq_len, _, _ = grid_repr.size()

        # 第一个预测器
        s = self.mlp2(F.gelu(grid_repr))
        o = self.mlp3(F.gelu(grid_repr))

        # 双仿射分类
        biaffine_scores = torch.einsum('bijh,hk,bikh->bijk', s, self.W, o) + \
                          torch.einsum('bijh,h->bij', s, self.b.unsqueeze(0)) + \
                          torch.einsum('bikh,h->bik', o, self.b.unsqueeze(0)).unsqueeze(1)

        # MLP分类
        mlp_scores = self.mlp1(grid_repr)

        # 组合两种分类方法的分数
        final_scores_1 = biaffine_scores + mlp_scores

        # 第二个预测器
        s_2 = self.mlp2_2(F.gelu(grid_repr))
        o_2 = self.mlp3_2(F.gelu(grid_repr))

        # 双仿射分类
        biaffine_scores_2 = torch.einsum('bijh,hk,bikh->bijk', s_2, self.W_2, o_2) + \
                            torch.einsum('bijh,h->bij', s_2, self.b_2.unsqueeze(0)) + \
                            torch.einsum('bikh,h->bik', o_2, self.b_2.unsqueeze(0)).unsqueeze(1)

        # MLP分类
        mlp_scores_2 = self.mlp1_2(grid_repr)

        # 组合两种分类方法的分数
        final_scores_2 = biaffine_scores_2 + mlp_scores_2

        # 计算两个预测器的概率分布
        probs_1 = F.softmax(final_scores_1, dim=-1)
        probs_2 = F.softmax(final_scores_2, dim=-1)

        return probs_1, probs_2


class TriggerRecognitionModule(nn.Module):
    """触发词识别分类模块"""

    def __init__(self, config):
        super(TriggerRecognitionModule, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # 条件层归一化
        self.cln = ConditionalLayerNorm(config.hidden_size)

        # 位置编码和区域编码
        self.position_embeddings = nn.Embedding(config.max_length * 2, config.hidden_size // 2)
        self.region_embeddings = nn.Embedding(3, config.hidden_size // 2)  # 3种区域类型: 上三角, 下三角和对角线

        # 混合嵌入的MLP
        self.mixer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # 不同膨胀率的卷积层
        self.dilated_convs = nn.ModuleList([
            DilatedConvolution(config.hidden_size, dilation=1),
            DilatedConvolution(config.hidden_size, dilation=2),
            DilatedConvolution(config.hidden_size, dilation=3)
        ])

        # 级联类型预测器 (考虑到实验中可能会移除此组件)
        if config.use_ctp:
            self.ctp = CascadeTypePredictor(config.hidden_size, len(self._get_event_types()))

    def _get_event_types(self):
        """获取事件类型"""
        # 这里应根据数据集动态获取，这里简化处理
        if self.config.dataset == "DUEE":
            return ["收购", "投资", "股权转让", ...]  # 实际应从配置或数据中加载
        # 处理其他数据集...

    def create_position_ids(self, seq_len):
        """创建相对位置编码ID"""
        position_ids = torch.zeros(seq_len, seq_len, dtype=torch.long)
        for i in range(seq_len):
            for j in range(seq_len):
                position_ids[i, j] = seq_len + j - i  # 相对位置编码
        return position_ids

    def create_region_ids(self, seq_len):
        """创建区域编码ID"""
        region_ids = torch.zeros(seq_len, seq_len, dtype=torch.long)
        for i in range(seq_len):
            for j in range(seq_len):
                if i < j:  # 上三角
                    region_ids[i, j] = 0
                elif i > j:  # 下三角
                    region_ids[i, j] = 1
                else:  # 对角线
                    region_ids[i, j] = 2
        return region_ids

    def forward(self, hidden_states):
        """
        输入: hidden_states [batch_size, seq_len, hidden_size]
        输出: trigger_probs [batch_size, seq_len, seq_len, num_event_types]
        """
        batch_size, seq_len, hidden_size = hidden_states.size()

        # 创建网格表示
        grid_repr = torch.zeros(batch_size, seq_len, seq_len, hidden_size).to(hidden_states.device)

        # 使用CLN生成网格表示
        for i in range(seq_len):
            for j in range(seq_len):
                grid_repr[:, i, j] = self.cln(hidden_states[:, i], hidden_states[:, j])

        # 获取位置编码和区域编码
        position_ids = self.create_position_ids(seq_len).to(hidden_states.device)
        region_ids = self.create_region_ids(seq_len).to(hidden_states.device)

        position_embeds = self.position_embeddings(position_ids).unsqueeze(0).expand(batch_size, -1, -1, -1)
        region_embeds = self.region_embeddings(region_ids).unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 合并网格表示、位置编码和区域编码
        combined_repr = torch.cat([grid_repr, position_embeds, region_embeds], dim=-1)
        grid_repr = self.mixer(combined_repr)

        # 应用空洞卷积
        conv_outputs = []
        for conv in self.dilated_convs:
            conv_outputs.append(conv(grid_repr))

        # 合并卷积输出
        grid_repr = sum(conv_outputs) / len(self.dilated_convs)

        # 应用级联类型预测器
        if self.config.use_ctp:
            trigger_probs_1, trigger_probs_2 = self.ctp(grid_repr)
            # 合并两个预测器的结果
            trigger_probs = torch.max(trigger_probs_1, trigger_probs_2)
        else:
            # 如果不使用CTP，则使用简单的线性层进行预测
            linear = nn.Linear(hidden_size, len(self._get_event_types())).to(hidden_states.device)
            trigger_probs = linear(grid_repr)
            trigger_probs = F.softmax(trigger_probs, dim=-1)

        return trigger_probs
