import torch
from torch.utils.data import Dataset, DataLoader


class EventExtractionDataset(Dataset):
    """事件抽取数据集"""

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        processed_sample = self.processor.process_sample(sample)
        return processed_sample


def collate_fn(batch):
    """
    批次数据整理函数
    """
    texts = [item["text"] for item in batch]
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    offset_mappings = [item["offset_mapping"] for item in batch]
    trigger_labels = torch.stack([item["trigger_labels"] for item in batch])
    argument_labels = torch.stack([item["argument_labels"] for item in batch])

    return {
        "texts": texts,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mappings": offset_mappings,
        "trigger_labels": trigger_labels,
        "argument_labels": argument_labels
    }


def get_data_loaders(config, processor):
    """
    获取数据加载器
    """
    # 加载数据
    train_data = processor.load_data("train")
    dev_data = processor.load_data("dev")
    test_data = processor.load_data("test")

    # 创建数据集
    train_dataset = EventExtractionDataset(train_data, processor)
    dev_dataset = EventExtractionDataset(dev_data, processor)
    test_dataset = EventExtractionDataset(test_data, processor)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, dev_loader, test_loader
