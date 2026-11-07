# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import random
from typing import List, Optional, Dict, Any

from torch.utils.data import Dataset

from .sam3_image_dataset import Sam3ImageDataset, Datapoint


class FewShotDataset(Dataset):
    """
    Few-Shot数据集包装器，支持从主数据集和多个support数据集中采样。
    
    主要特点：
    1. 使用主数据集进行训练（main_dataset）
    2. 支持多个support数据集（support_datasets）
    3. 在__init__中固定采样num_support_shots个支持样本
    4. 每个item返回{'query': query, 'fixed_support_pool': pool}
    5. support样本会带上is_support标记
    6. __len__只返回主数据集的长度（保证step数不变）
    """
    
    def __init__(
        self,
        main_dataset: Sam3ImageDataset,
        support_datasets: Optional[List[Sam3ImageDataset]] = None,
        num_support_shots: int = 1,
        enable_few_shot: bool = True,
    ):
        """
        Args:
            main_dataset: 主训练数据集（已实例化）
            support_datasets: Support数据集列表（已实例化，可选）
            num_support_shots: 固定采样的支持样本数
            enable_few_shot: 是否启用few-shot
        """
        super().__init__()
        
        self.main_dataset = main_dataset
        self.support_datasets = support_datasets or []
        self.num_support_shots = num_support_shots
        self.enable_few_shot = enable_few_shot
        
        # 在__init__中就固定采样好支持样本
        self._sample_fixed_support_pool()
    
    def _sample_fixed_support_pool(self):
        """在初始化时就固定采样好num_support_shots个支持样本"""
        self.fixed_support_pool = []
        
        if not self.enable_few_shot or len(self.support_datasets) == 0:
            return
        
        # 构建所有支持数据集的样本池
        all_support_samples = []
        for dataset in self.support_datasets:
            for idx in range(len(dataset)):
                all_support_samples.append({
                    'dataset': dataset,
                    'index': idx
                })
        
        # 随机采样固定数量的支持样本
        num_samples = min(self.num_support_shots, len(all_support_samples))
        sampled_indices = random.sample(range(len(all_support_samples)), num_samples)
        
        # 加载这些支持样本，保存到固定池中，并加上is_support标记
        for i in sampled_indices:
            item_info = all_support_samples[i]
            try:
                support_dp = item_info['dataset'][item_info['index']]
                # 给support样本加标记
                support_dp.is_support = True
                self.fixed_support_pool.append(support_dp)
            except Exception as e:
                print(f"Warning: Error loading support image {item_info['index']}: {e}")
    
    def __getitem__(self, idx):
        if not self.enable_few_shot or len(self.fixed_support_pool) == 0:
            return self.main_dataset[idx]
        
        # 获取查询样本，标记为非support
        query_datapoint = self.main_dataset[idx]
        query_datapoint.is_support = False
        
        # 返回包含query和支持样本池的字典
        return {
            'query': query_datapoint,
            'fixed_support_pool': self.fixed_support_pool,
        }
    
    def __len__(self) -> int:
        """只返回主数据集的长度，保证step数不变"""
        return len(self.main_dataset)


def build_few_shot_dataset(
    main_dataset,
    support_datasets=None,
    num_support_shots: int = 1,
    enable_few_shot: bool = True,
    **kwargs,
) -> FewShotDataset:
    """
    构建Few-Shot数据集的工厂函数。
    
    Args:
        main_dataset: 主数据集（已实例化好的 Sam3ImageDataset）
        support_datasets: Support数据集列表（已实例化好的）
        num_support_shots: 固定采样的支持样本数
        enable_few_shot: 是否启用few-shot
        **kwargs: 忽略的其他参数
    
    Returns:
        FewShotDataset实例
    """
    return FewShotDataset(
        main_dataset=main_dataset,
        support_datasets=support_datasets,
        num_support_shots=num_support_shots,
        enable_few_shot=enable_few_shot,
    )
