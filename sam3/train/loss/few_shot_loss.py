
# Few-Shot Learning 专用损失模块
# 包含原型对齐损失、对比学习损失等

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_fns import CORE_LOSS_KEY


def _normalize_concept_name(name: str):
    return name.strip().lower().replace(" ", "_")


class PrototypeAlignmentLoss(nn.Module):
    """
    原型对齐损失：将同概念的 query 特征与 support 原型拉近
    """

    def __init__(
        self,
        weight: float = 0.2,
        distance_type: str = "mse",
        temperature: float = 1.0,
    ):
        super(PrototypeAlignmentLoss, self).__init__()
        self.weight = float(weight)
        self.distance_type = distance_type
        self.temperature = float(temperature)

    def forward(
        self,
        features: torch.Tensor,
        concept_names: List[str],
        is_support_mask: torch.Tensor,
    ):
        """
        计算原型对齐损失
        支持双向拉近：两张图可以相互作为 query 和 support

        Args:
            features: (num_samples, hidden_dim) 所有样本特征（包含 query 和 support）
            concept_names: 所有样本的概念名称列表
            is_support_mask: (num_samples,) bool 掩码，标记哪些是 support

        Returns:
            loss: 标量损失
        """
        if not is_support_mask.any() or is_support_mask.all():
            return torch.tensor(0.0, device=features.device)

        device = features.device
        dtype = features.dtype

        # 双向拉近：计算两次损失
        # 第一次：原图的 is_support_mask（图1是support，图0是query）
        # 第二次：反转 is_support_mask（图0是support，图1是query）
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_count = 0

        # 定义两个方向的掩码
        support_masks = [
            is_support_mask,  # 方向1：原图
            ~is_support_mask  # 方向2：反转
        ]

        for current_support_mask in support_masks:
            # 按概念分组
            concept_to_features = defaultdict(list)
            concept_to_support_features = defaultdict(list)
            for feat, name, is_support in zip(features, concept_names, current_support_mask):
                normalized_name = _normalize_concept_name(name)
                concept_to_features[normalized_name].append(feat)
                if is_support:
                    concept_to_support_features[normalized_name].append(feat)

            loss = torch.tensor(0.0, device=device, dtype=dtype)
            count = 0

            for concept_name, feats in concept_to_features.items():
                # 检查是否有 support 样本
                if concept_name not in concept_to_support_features:
                    continue
                
                support_feats = concept_to_support_features[concept_name]
                if len(support_feats) < 1:
                    continue
                
                support_feats_tensor = torch.stack(support_feats, dim=0)
                
                # 只使用当前方向的 support 样本计算原型
                prototype = support_feats_tensor.mean(dim=0)
                
                # 只对当前方向的 query 样本计算到原型的距离
                query_feats = []
                for feat, is_support in zip(feats, current_support_mask):
                    if not is_support:
                        query_feats.append(feat)
                
                if len(query_feats) < 1:
                    continue
                
                query_feats_tensor = torch.stack(query_feats, dim=0)
                
                # 计算所有 query 样本到原型的距离
                if self.distance_type == "mse":
                    dist = F.mse_loss(query_feats_tensor, prototype.expand_as(query_feats_tensor), reduction="none")
                    loss = loss + dist.sum()
                elif self.distance_type == "cosine":
                    feats_norm = F.normalize(query_feats_tensor, p=2, dim=-1)
                    proto_norm = F.normalize(prototype, p=2, dim=-1)
                    sim = (feats_norm * proto_norm.unsqueeze(0)).sum(dim=-1)
                    loss = loss + (1 - sim).sum()
                
                count += len(query_feats)

            if count > 0:
                loss = loss / count
                total_loss = total_loss + loss
                total_count += count

        # 平均两个方向的损失
        if total_count > 0:
            total_loss = total_loss / 2

        return total_loss


class FewShotContrastiveLoss(nn.Module):
    """
    Few-Shot 对比学习损失：
    - 同概念样本在隐空间中拉近
    - 不同概念样本在隐空间中推开
    """

    def __init__(
        self,
        weight: float = 0.1,
        temperature: float = 0.5,
    ):
        super(FewShotContrastiveLoss, self).__init__()
        self.weight = float(weight)
        self.temperature = float(temperature)

    def forward(
        self,
        features: torch.Tensor,
        concept_names: List[str],
        is_support_mask: Optional[torch.Tensor] = None,
    ):
        """
        计算对比学习损失（完整 InfoNCE Loss）
        优化版本：完全向量化，避免 Python 循环

        Args:
            features: (N, hidden_dim) 所有样本特征
            concept_names: 长度为 N 的概念名称列表
            is_support_mask: (N,) 可选，标记哪些是 support 样本

        Returns:
            loss: 标量损失
        """
        N = features.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=features.device)

        device = features.device
        dtype = features.dtype

        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=-1)

        # 计算相似度矩阵
        sim_matrix = features_norm @ features_norm.t() / self.temperature

        # 构建概念名称到索引的映射
        name_to_indices = defaultdict(list)
        for i, name in enumerate(concept_names):
            name_to_indices[_normalize_concept_name(name)].append(i)

        # 构建正样本掩码矩阵 (N, N)
        pos_mask = torch.zeros(N, N, device=device, dtype=torch.bool)
        for concept_name, indices in name_to_indices.items():
            if len(indices) >= 2:
                indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                pos_mask[indices_tensor[:, None], indices_tensor[None, :]] = True

        # 排除自己和自己比较
        eye_mask = torch.eye(N, device=device, dtype=torch.bool)
        pos_mask = pos_mask & ~eye_mask

        # 找到有正样本的 anchor
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device)

        # 只处理有正样本的 anchor
        valid_indices = torch.where(has_pos)[0]
        sim_valid = sim_matrix[valid_indices]
        pos_mask_valid = pos_mask[valid_indices]

        # 减去对角线的最大值（数值稳定性）
        logits_max, _ = sim_valid.max(dim=1, keepdim=True)
        logits = sim_valid - logits_max.detach()

        # 计算 exp(logits)
        exp_logits = logits.exp()

        # 计算每行的正样本和
        pos_sum = (exp_logits * pos_mask_valid.float()).sum(dim=1, keepdim=True)

        # 计算每行的所有样本和（除了自己）
        neg_mask_valid = ~eye_mask[valid_indices]
        all_sum = (exp_logits * neg_mask_valid.float()).sum(dim=1, keepdim=True)

        # 计算 log(pos_sum / all_sum)
        log_prob = (pos_sum / all_sum).log()

        # 对每个正样本取平均
        num_pos_per_row = pos_mask_valid.float().sum(dim=1, keepdim=True)
        log_prob = log_prob / num_pos_per_row

        # 计算损失
        loss = -log_prob.mean()

        return loss


class FewShotAuxLoss(nn.Module):
    """
    Few-Shot 综合辅助损失：
    - 原型对齐损失
    - 对比学习损失
    """

    def __init__(
        self,
        prototype_alignment_weight: float = 0.2,
        contrastive_weight: float = 0.1,
        prototype_distance_type: str = "mse",
        contrastive_temperature: float = 0.5,
    ):
        super().__init__()
        self.prototype_loss = PrototypeAlignmentLoss(
            weight=prototype_alignment_weight,
            distance_type=prototype_distance_type,
        )
        self.contrastive_loss = FewShotContrastiveLoss(
            weight=contrastive_weight,
            temperature=contrastive_temperature,
        )
        self.prototype_alignment_weight = float(prototype_alignment_weight)
        self.contrastive_weight = float(contrastive_weight)

    def forward(
        self,
        outputs,
        targets,
        indices=None,
        num_boxes=None,
        is_aux=False,
    ):
        if is_aux:
            device = outputs.get("pred_logits", torch.zeros(1)).device
            return {CORE_LOSS_KEY: torch.tensor(0.0, device=device)}

        device = outputs.get("pred_logits", torch.zeros(1)).device
        dtype = outputs.get("pred_logits", torch.zeros(1)).dtype

        loss_proto = torch.tensor(0.0, device=device, dtype=dtype)
        loss_contrastive = torch.tensor(0.0, device=device, dtype=dtype)

        # 检查是否有 Few-Shot 相关信息
        mask_embeddings = outputs.get("mask_embeddings")
        is_support_mask = outputs.get("is_support_mask")
        prompt_names = outputs.get("find_prompt_names")

        if mask_embeddings is not None and is_support_mask is not None and prompt_names:
            if is_support_mask.any() and not is_support_mask.all():
                loss_proto = self.prototype_loss(
                    features=mask_embeddings,
                    concept_names=prompt_names,
                    is_support_mask=is_support_mask,
                )

                loss_contrastive = self.contrastive_loss(
                    features=mask_embeddings,
                    concept_names=prompt_names,
                    is_support_mask=is_support_mask,
                )

        total = (
            self.prototype_alignment_weight * loss_proto
            + self.contrastive_weight * loss_contrastive
        )

        return {
            CORE_LOSS_KEY: total,
            "loss_few_shot_proto": loss_proto,
            "loss_few_shot_contrastive": loss_contrastive,
        }

