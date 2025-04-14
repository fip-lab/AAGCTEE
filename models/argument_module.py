import torch
import torch.nn as nn
import torch.nn.functional as F


class ArgumentAssociationGraph(nn.Module):
    """论元关联图"""

    def __init__(self, config):
        super(ArgumentAssociationGraph, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # 用于论元实体向量变换的MLP
        self.entity_mlp1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.entity_mlp2 = nn.Linear(config.hidden_size, config.hidden_size)

        # 可学习的关联阈值
        self.threshold = nn.Parameter(torch.tensor(0.5))

        # 图卷积网络层
        self.gcn_layers = nn.ModuleList([
            GraphConvolutionLayer(config.hidden_size) for _ in range(2)  # 2层GCN
        ])

    def forward(self, entity_repr):
        """
        构建论元关联图并更新实体表示

        entity_repr: [batch_size, num_entities, hidden_size] 实体表示
        """
        batch_size, num_entities, _ = entity_repr.size()

        # 变换实体表示用于计算关联强度
        entity_repr_1 = self.entity_mlp1(entity_repr)
        entity_repr_2 = self.entity_mlp2(entity_repr)

        # 计算实体间的关联强度
        association_scores = torch.bmm(
            entity_repr_1,
            entity_repr_2.transpose(1, 2)
        ) / (self.hidden_size ** 0.5)  # [batch_size, num_entities, num_entities]

        # 构建邻接矩阵
        adjacency = (association_scores > self.threshold).float()

        # 构建度矩阵 (对角矩阵)
        degree = torch.sum(adjacency, dim=-1)  # [batch_size, num_entities]
        degree_matrix = torch.diag_embed(torch.pow(degree + 1e-8, -0.5))  # [batch_size, num_entities, num_entities]

        # 应用图卷积网络更新实体表示
        normalized_adj = torch.bmm(torch.bmm(degree_matrix, adjacency), degree_matrix)

        updated_entity_repr = entity_repr
        for gcn_layer in self.gcn_layers:
            updated_entity_repr = gcn_layer(updated_entity_repr, normalized_adj)

        return updated_entity_repr, association_scores


class GraphConvolutionLayer(nn.Module):
    """图卷积层"""

    def __init__(self, hidden_size):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        x: [batch_size, num_nodes, hidden_size] 节点特征
        adj: [batch_size, num_nodes, num_nodes] 邻接矩阵
        """
        support = torch.bmm(x, self.weight.unsqueeze(0).expand(x.size(0), -1, -1))
        output = torch.bmm(adj, support)
        return F.relu(output + self.bias)


class GlobalNormalizationDecoding(nn.Module):
    """全局归一化解码策略"""

    def __init__(self, config, num_event_types, num_role_types):
        super(GlobalNormalizationDecoding, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_event_types = num_event_types
        self.num_role_types = num_role_types

        # 用于论元角色预测的查询和键向量变换
        self.query_transforms = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(num_role_types)
            ]) for _ in range(num_event_types)
        ])

        self.key_transforms = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(num_role_types)
            ]) for _ in range(num_event_types)
        ])

        # 旋转位置编码
        self.rotary_pos_emb = RotaryPositionalEmbedding(config.hidden_size)

        # 学习阈值的参数
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, entity_repr, association_scores, event_types):
        """
        全局归一化解码

        hidden_states: [batch_size, seq_len, hidden_size] 序列表示
        entity_repr: [batch_size, num_entities, hidden_size] 实体表示
        association_scores: [batch_size, num_entities, num_entities] 实体间关联强度
        event_types: [batch_size, num_events] 每个样本的事件类型

        返回: [batch_size, num_event_types, num_role_types, seq_len, seq_len] 论元角色概率
        """
        batch_size, seq_len, _ = hidden_states.size()

        # 初始化输出张量
        arg_scores = torch.zeros(
            batch_size, self.num_event_types, self.num_role_types,
            seq_len, seq_len
        ).to(hidden_states.device)

        # 对每个事件类型和角色类型计算分数
        for event_type_idx in range(self.num_event_types):
            for role_type_idx in range(self.num_role_types):
                # 获取当前事件类型和角色类型的变换
                query_transform = self.query_transforms[event_type_idx][role_type_idx]
                key_transform = self.key_transforms[event_type_idx][role_type_idx]

                # 计算序列中每对位置的查询和键向量
                for i in range(seq_len):
                    for j in range(i, seq_len):  # 只计算i<=j的部分
                        # 应用旋转位置编码
                        q_i = query_transform(hidden_states[:, i])
                        k_j = key_transform(hidden_states[:, j])

                        q_i_rot, k_j_rot = self.rotary_pos_emb(q_i, k_j, i, j)

                        # 计算论元角色分数
                        score = torch.sum(q_i_rot * k_j_rot, dim=-1)  # [batch_size]

                        # 如果使用论元关联图
                        if self.config.use_aag and association_scores is not None:
                            # 找到对应的实体索引
                            entity_i_idx = i  # 简化处理，实际应根据实体边界查找
                            entity_j_idx = j

                            if entity_i_idx < association_scores.size(1) and entity_j_idx < association_scores.size(2):
                                # 结合实体关联得分
                                entity_assoc = association_scores[:, entity_i_idx, entity_j_idx]
                                score = score + entity_assoc

                        arg_scores[:, event_type_idx, role_type_idx, i, j] = score

        return arg_scores


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码"""

    def __init__(self, hidden_size):
        super(RotaryPositionalEmbedding, self).__init__()
        self.hidden_size = hidden_size

        # 初始化旋转矩阵参数
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k, pos_i, pos_j):
        """
        应用旋转位置编码

        q: [batch_size, hidden_size] 查询向量
        k: [batch_size, hidden_size] 键向量
        pos_i, pos_j: 位置索引
        """
        # 计算相对位置
        rel_pos = pos_j - pos_i

        # 生成旋转矩阵
        device = q.device
        if self.cos_cached is None or self.sin_cached is None:
            self._generate_rotation_matrix(device)

        # 获取对应位置的旋转矩阵
        cos_pos = self.cos_cached[rel_pos].to(device)
        sin_pos = self.sin_cached[rel_pos].to(device)

        # 应用旋转变换
        q_rot = (q * cos_pos) + (self._rotate_half(q) * sin_pos)
        k_rot = (k * cos_pos) + (self._rotate_half(k) * sin_pos)

        return q_rot, k_rot

    def _generate_rotation_matrix(self, device):
        """生成旋转矩阵"""
        max_length = 512  # 足够大的上下文长度

        # 初始化旋转矩阵缓存
        self.cos_cached = torch.zeros(max_length * 2, self.hidden_size).to(device)
        self.sin_cached = torch.zeros(max_length * 2, self.hidden_size).to(device)

        # 生成旋转矩阵
        for pos in range(-max_length, max_length):
            idx = pos + max_length
            for i in range(0, self.hidden_size, 2):
                theta = 10000 ** (-2 * (i // 2) / self.hidden_size)
                self.cos_cached[idx, i:i + 2] = torch.cos(pos * theta)
                self.sin_cached[idx, i:i + 2] = torch.sin(pos * theta)

    def _rotate_half(self, x):
        """旋转一半的维度"""
        x1, x2 = x[..., :self.hidden_size // 2], x[..., self.hidden_size // 2:]
        return torch.cat((-x2, x1), dim=-1)


class ArgumentExtractionModule(nn.Module):
    """论元识别分类模块"""

    def __init__(self, config, num_event_types, num_role_types):
        super(ArgumentExtractionModule, self).__init__()
        self.config = config
        self.num_event_types = num_event_types
        self.num_role_types = num_role_types

        # 论元关联图
        if config.use_aag:
            self.aag = ArgumentAssociationGraph(config)

        # 全局归一化解码
        if config.use_gnd:
            self.gnd = GlobalNormalizationDecoding(config, num_event_types, num_role_types)
        else:
            # 简单的分类器
            self.classifier = nn.Linear(config.hidden_size * 2, num_role_types)

    def forward(self, hidden_states, trigger_spans, event_types):
        """
        论元识别分类前向传播

        hidden_states: [batch_size, seq_len, hidden_size] 序列表示
        trigger_spans: [(start_idx, end_idx), ...] 触发词span列表
        event_types: [batch_size, num_events] 每个样本的事件类型

        返回: [batch_size, num_event_types, num_role_types, seq_len, seq_len] 论元角色概率
        """
        batch_size, seq_len, hidden_size = hidden_states.size()

        # 提取实体表示 (简化处理，实际应根据命名实体识别结果)
        # 这里假设每个位置都是潜在的实体
        entity_repr = hidden_states

        # 使用论元关联图更新实体表示
        association_scores = None
        if self.config.use_aag:
            entity_repr, association_scores = self.aag(entity_repr)

        # 使用全局归一化解码进行论元角色识别
        if self.config.use_gnd:
            arg_scores = self.gnd(hidden_states, entity_repr, association_scores, event_types)
        else:
            # 简单的分类方法
            arg_scores = torch.zeros(
                batch_size, self.num_event_types, self.num_role_types,
                seq_len, seq_len
            ).to(hidden_states.device)

            for i in range(seq_len):
                for j in range(i, seq_len):
                    # 拼接两个位置的表示
                    pair_repr = torch.cat([hidden_states[:, i], hidden_states[:, j]], dim=-1)
                    # 应用分类器
                    scores = self.classifier(pair_repr)  # [batch_size, num_role_types]

                    # 对每个事件类型重复相同的分数
                    for event_type_idx in range(self.num_event_types):
                        arg_scores[:, event_type_idx, :, i, j] = scores

        # 计算多标签分类损失
        arg_probs = torch.sigmoid(arg_scores)

        return arg_probs
