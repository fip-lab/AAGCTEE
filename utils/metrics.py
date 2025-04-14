import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support


def compute_metrics(predictions, labels, event_types, role_types):
    """
    计算事件抽取评估指标

    predictions: 预测结果
    labels: 真实标签
    event_types: 事件类型列表
    role_types: 论元角色类型列表

    返回: 各项评估指标
    """
    results = {
        'TI': defaultdict(list),  # 触发词识别
        'TC': defaultdict(list),  # 触发词分类
        'AI': defaultdict(list),  # 论元识别
        'AC': defaultdict(list)  # 论元分类
    }

    # 处理每个样本
    for pred, label in zip(predictions, labels):
        # 触发词识别评估
        ti_pred = [(s, e) for e_type, s, e in pred['triggers']]
        ti_true = [(s, e) for e_type, s, e in label['triggers']]

        results['TI']['precision'].append(precision(ti_pred, ti_true))
        results['TI']['recall'].append(recall(ti_pred, ti_true))
        results['TI']['f1'].append(f1_score(ti_pred, ti_true))

        # 触发词分类评估
        tc_pred = [(e_type, s, e) for e_type, s, e in pred['triggers']]
        tc_true = [(e_type, s, e) for e_type, s, e in label['triggers']]

        results['TC']['precision'].append(precision(tc_pred, tc_true))
        results['TC']['recall'].append(recall(tc_pred, tc_true))
        results['TC']['f1'].append(f1_score(tc_pred, tc_true))

        # 论元识别评估
        ai_pred = [(s, e) for _, s, e, _ in pred['arguments']]
        ai_true = [(s, e) for _, s, e, _ in label['arguments']]

        results['AI']['precision'].append(precision(ai_pred, ai_true))
        results['AI']['recall'].append(recall(ai_pred, ai_true))
        results['AI']['f1'].append(f1_score(ai_pred, ai_true))

        # 论元分类评估
        ac_pred = [(e_type, r_type, s, e) for e_type, s, e, r_type in pred['arguments']]
        ac_true = [(e_type, r_type, s, e) for e_type, s, e, r_type in label['arguments']]

        results['AC']['precision'].append(precision(ac_pred, ac_true))
        results['AC']['recall'].append(recall(ac_pred, ac_true))
        results['AC']['f1'].append(f1_score(ac_pred, ac_true))

    # 计算平均值
    metrics = {}
    for task in ['TI', 'TC', 'AI', 'AC']:
        for metric in ['precision', 'recall', 'f1']:
            metrics[f'{task}_{metric}'] = np.mean(results[task][metric]) * 100

    # 计算平均F1
    metrics['Avg_F1'] = np.mean([metrics['TI_f1'], metrics['TC_f1'], metrics['AI_f1'], metrics['AC_f1']])

    return metrics


def precision(pred, true):
    """计算精确率"""
    if not pred:
        return 1.0 if not true else 0.0
    return len(set(pred) & set(true)) / len(pred)


def recall(pred, true):
    """计算召回率"""
    if not true:
        return 1.0 if not pred else 0.0
    return len(set(pred) & set(true)) / len(true)


def f1_score(pred, true):
    """计算F1值"""
    p = precision(pred, true)
    r = recall(pred, true)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
