import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KnowledgeFusionModule(nn.Module):
    """
    GNN 融合知识图谱信息：当前概念 + 相邻概念 + 边信息
    用于在输入阶段增强 prompt
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # GNN 层 1：融合当前、邻居、边
        self.gnn1 = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # GNN 层 2：进一步融合
        self.gnn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        current_emb: torch.Tensor,      # (num_concepts, embed_dim) - 当前概念
        neighbor_emb: torch.Tensor,     # (num_concepts, embed_dim) - 邻居概念均值
        edge_emb: torch.Tensor,         # (num_concepts, embed_dim) - 边信息
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回融合后的 embedding 和 dummy attention weights
        """
        device = current_emb.device

        # 拼接三个特征
        combined = torch.cat([current_emb, neighbor_emb, edge_emb], dim=-1)

        # GNN 层 1
        fused = self.gnn1(combined)
        fused = self.norm1(fused + current_emb)

        # GNN 层 2
        fused = self.gnn2(fused)
        fused = self.norm2(fused)

        # dummy attention weights（占位）
        attn_weights = torch.zeros(
            fused.shape[0], fused.shape[0],
            device=device, dtype=fused.dtype
        )

        return fused, attn_weights


def _normalize_concept_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


class MedicalKnowledgeBase(nn.Module):
    """
    医学知识图谱：
    - 节点 embedding：由视觉特征 momentum 更新
    - 边 embedding：由文本特征 momentum 更新（代表关系）
    - 静态更新，无梯度
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_concepts: int = 64,
        num_edges: int = 128,
        knowledge_config_path: Optional[str] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.structure_graph: Dict[str, List[str]] = {}
        self.structure_descriptions: Dict[str, Dict[str, List[str]]] = {}
        self._concept_to_idx: Dict[str, int] = {}
        self._edge_list: List[Tuple[str, str]] = []

        if knowledge_config_path and os.path.isfile(knowledge_config_path):
            with open(knowledge_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.structure_graph = data.get("structure_graph", {})
            self.structure_descriptions = data.get("structure_descriptions", {})
            self._init_graph_from_structure()

        num_real_concepts = len(self._concept_to_idx)
        if num_real_concepts == 0:
            num_real_concepts = num_concepts
        num_real_edges = len(self._edge_list)
        if num_real_edges == 0:
            num_real_edges = num_edges

        self.register_buffer("concept_embeddings", torch.zeros(num_real_concepts, embed_dim))
        self.register_buffer("edge_embeddings", torch.zeros(num_real_edges, embed_dim))

        self.knowledge_fusion = KnowledgeFusionModule(embed_dim)

    def _init_graph_from_structure(self):
        # 收集所有出现的概念名（包括 keys 和 values）
        all_concepts = set()
        for a, neighbors in self.structure_graph.items():
            all_concepts.add(a)
            for b in neighbors:
                all_concepts.add(b)
        
        # 也加入 structure_descriptions 中的概念
        for concept_name in self.structure_descriptions.keys():
            all_concepts.add(concept_name)
        
        concept_names = sorted(all_concepts)
        self._concept_to_idx = {_normalize_concept_name(n): i for i, n in enumerate(concept_names)}
        
        # 构建边列表
        seen = set()
        for a in self.structure_graph.keys():
            a_key = _normalize_concept_name(a)
            for b in self.structure_graph.get(a, []):
                b_key = _normalize_concept_name(b)
                if a_key == b_key:
                    continue
                pair = (min(a_key, b_key), max(a_key, b_key))
                if pair not in seen:
                    seen.add(pair)
                    self._edge_list.append(pair)

    def get_embeddings_for_prompts(
        self, prompt_names: List[str], device: torch.device
    ) -> torch.Tensor:
        """获取当前 batch 概念的 embedding"""
        embeddings = []
        for name in prompt_names:
            key = _normalize_concept_name(name)
            idx = self._concept_to_idx.get(key, 0)
            if idx < self.concept_embeddings.shape[0]:
                embeddings.append(self.concept_embeddings[idx])
            else:
                embeddings.append(torch.zeros(self.embed_dim, device=device))
        return torch.stack(embeddings, dim=0).to(device)

    def get_neighbor_embeddings(
        self, prompt_names: List[str], device: torch.device
    ) -> torch.Tensor:
        """获取当前 batch 概念的邻居 embedding 均值"""
        neighbor_embs = []
        for name in prompt_names:
            key = _normalize_concept_name(name)
            neighbors = self.structure_graph.get(name, [])
            
            embs = []
            for nb in neighbors:
                nb_key = _normalize_concept_name(nb)
                nb_idx = self._concept_to_idx.get(nb_key, -1)
                if nb_idx >= 0 and nb_idx < self.concept_embeddings.shape[0]:
                    embs.append(self.concept_embeddings[nb_idx])
            
            if embs:
                neighbor_embs.append(torch.stack(embs, dim=0).mean(dim=0))
            else:
                neighbor_embs.append(torch.zeros(self.embed_dim, device=device))
        
        return torch.stack(neighbor_embs, dim=0).to(device)

    def get_edge_embeddings_for_prompts(
        self, prompt_names: List[str], device: torch.device
    ) -> torch.Tensor:
        """获取当前 batch 概念的边 embedding"""
        edge_embs = []
        for name in prompt_names:
            key = _normalize_concept_name(name)
            found_edge = False
            
            for idx, (a, b) in enumerate(self._edge_list):
                if idx >= self.edge_embeddings.shape[0]:
                    continue
                if a == key or b == key:
                    edge_embs.append(self.edge_embeddings[idx])
                    found_edge = True
                    break
            
            if not found_edge:
                edge_embs.append(torch.zeros(self.embed_dim, device=device))
        
        return torch.stack(edge_embs, dim=0).to(device)

    def fuse_and_update_knowledge(
        self,
        visual_emb: torch.Tensor,
        text_emb: torch.Tensor,
        prompt_names: List[str],
        momentum: float = 0.9,
        node_visual_weight: float = 0.7,
        node_text_weight: float = 0.3,
    ) -> None:
        """
        静态更新知识图谱：只负责更新，不做融合
        - 节点：视觉特征 + 文本特征融合后 momentum 更新
        - 边：文本特征对称差值后 momentum 更新
        
        Args:
            visual_emb: (num_prompts, embed_dim) - 当前样本的视觉特征
            text_emb: (num_prompts, embed_dim) - 文本特征
            prompt_names: 概念名称列表
            momentum: 动量更新系数
            node_visual_weight: 节点更新时视觉特征的权重
            node_text_weight: 节点更新时文本特征的权重
        """
        name_to_visual = {}
        name_to_text = {}
        for i, name in enumerate(prompt_names):
            key = _normalize_concept_name(name)
            name_to_visual[key] = visual_emb[i]
            name_to_text[key] = text_emb[i]
        
        with torch.no_grad():
            # 更新节点：融合视觉和文本
            for name, idx in self._concept_to_idx.items():
                if name in name_to_visual and name in name_to_text and idx < self.concept_embeddings.shape[0]:
                    combined_emb = (
                        node_visual_weight * name_to_visual[name] +
                        node_text_weight * name_to_text[name]
                    )
                    self.concept_embeddings.data[idx] = (
                        momentum * self.concept_embeddings.data[idx] +
                        (1 - momentum) * combined_emb
                    )
            
            # 更新边：对称差值
            for idx, (a, b) in enumerate(self._edge_list):
                if idx >= self.edge_embeddings.shape[0]:
                    continue
                if a in name_to_text and b in name_to_text:
                    # 对称化：使用绝对值，保证无向边的一致性
                    diff = name_to_text[a] - name_to_text[b]
                    new_edge_emb = torch.abs(diff)
                    self.edge_embeddings.data[idx] = (
                        momentum * self.edge_embeddings.data[idx] +
                        (1 - momentum) * new_edge_emb
                    )

    def fuse_knowledge_for_prompt(
        self,
        prompt_names: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        用 GNN 融合知识图谱信息，用于 prompt 增强
        
        Args:
            prompt_names: 概念名称列表
            device: 设备
        
        Returns:
            fused_emb: (num_prompts, embed_dim) - 融合后的 embedding
        """
        current_emb = self.get_embeddings_for_prompts(prompt_names, device)
        neighbor_emb = self.get_neighbor_embeddings(prompt_names, device)
        edge_emb = self.get_edge_embeddings_for_prompts(prompt_names, device)
        
        with torch.no_grad():
            fused, _ = self.knowledge_fusion(current_emb, neighbor_emb, edge_emb)
        
        return fused

    def get_rich_prompt(self, concept_name: str) -> str:
        key = _normalize_concept_name(concept_name)
        descriptions = self.structure_descriptions.get(concept_name, {})
        
        prompt_parts = [concept_name]
        
        # 添加形态学描述（第一条）
        morphology = descriptions.get("morphology", [])
        if morphology:
            prompt_parts.append(f"- {morphology[0]}")
        
        # 添加关系描述（第一条）
        relation = descriptions.get("relation", [])
        if relation:
            prompt_parts.append(f"- {relation[0]}")
        
        # 添加邻居
        neighbors = self.structure_graph.get(concept_name, [])
        if neighbors:
            prompt_parts.append(f"Related: {', '.join(neighbors)}")
        
        return " ".join(prompt_parts)

