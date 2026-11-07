import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model import box_ops
from sam3.model.geometric_reasoning import GeometricReasoner
from .loss_fns import CORE_LOSS_KEY
from .few_shot_loss import FewShotAuxLoss


MAIN_STRUCTURES = {"left_ventricle", "left_atrium", "right_ventricle", "right_atrium"}


class MedicalAuxLoss(nn.Module):
    """
    MedicalAuxLoss: 生成 pseudo labels 后计算 loss。
    
    核心思路：
    1. 推理 pseudo mask/bbox (几何推理)
    2. 为每个 pseudo target 选择最佳 query（使用与 GT 一致的成本函数）
    3. 计算与 GT 一致的 loss
    
    注意：Pseudo targets 对应特定 batch_idx，不需要匈牙利匹配。
    """

    def __init__(
        self,
        knowledge_config_path: Optional[str] = None,
        use_few_shot_loss: bool = False,
        few_shot_proto_weight: float = 0.2,
        few_shot_contrastive_weight: float = 0.1,
        use_pseudo_labels: bool = False,
        pseudo_label_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
        loss_mask_weight: float = 1.0,
        loss_cls_weight: float = 1.0,
    ):
        super().__init__()
        
        self.knowledge_config = {}
        if knowledge_config_path and os.path.isfile(knowledge_config_path):
            with open(knowledge_config_path, "r", encoding="utf-8") as f:
                self.knowledge_config = json.load(f)

        self.use_few_shot_loss = use_few_shot_loss
        self.few_shot_loss = None
        if self.use_few_shot_loss:
            self.few_shot_loss = FewShotAuxLoss(
                prototype_alignment_weight=few_shot_proto_weight,
                contrastive_weight=few_shot_contrastive_weight,
            )

        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_label_weight = pseudo_label_weight
        self.geometric_reasoner = GeometricReasoner() if self.use_pseudo_labels else None
        
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_cls_weight = loss_cls_weight

    def _mask_to_bbox(self, mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """将 mask 转换为归一化的 xyxy bbox"""
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        
        if not rows.any() or not cols.any():
            return torch.tensor([0.0, 0.0, 1.0, 1.0], device=mask.device, dtype=torch.float32)
        
        y_indices = torch.where(rows)[0]
        x_indices = torch.where(cols)[0]
        
        y1, y2 = y_indices[0].item(), y_indices[-1].item() + 1
        x1, x2 = x_indices[0].item(), x_indices[-1].item() + 1
        
        return torch.tensor([x1 / W, y1 / H, x2 / W, y2 / H], device=mask.device, dtype=torch.float32)

    def _infer_pseudo_structures(
        self,
        gt_masks: Dict[str, torch.Tensor],
        gt_bboxes_xyxy: Dict[str, torch.Tensor],
        H: int,
        W: int,
        device: torch.device,
    ) -> Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """推理所有伪结构"""
        inferred: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        
        lv_mask = gt_masks.get("left_ventricle")
        rv_mask = gt_masks.get("right_ventricle")
        la_mask = gt_masks.get("left_atrium")
        ra_mask = gt_masks.get("right_atrium")
        
        rv_bbox = gt_bboxes_xyxy.get("right_ventricle")
        ra_bbox = gt_bboxes_xyxy.get("right_atrium")
        la_bbox = gt_bboxes_xyxy.get("left_atrium")
        
        if lv_mask is None and la_mask is not None and rv_mask is not None:
            if la_bbox is not None or rv_bbox is not None:
                inferred_lv_bbox = self.geometric_reasoner.infer_lv(
                    rv_bbox=rv_bbox, ra_bbox=ra_bbox, la_bbox=la_bbox,
                    image_width=1.0, use_random=True,
                )
                if inferred_lv_bbox is not None:
                    inferred_lv_mask = self.geometric_reasoner.generate_ellipse_mask_from_bbox(
                        inferred_lv_bbox, H, W, use_random=True
                    )
                    if inferred_lv_mask is not None:
                        inferred["left_ventricle"] = (inferred_lv_mask, inferred_lv_bbox)
                        lv_mask = inferred_lv_mask
        
        if lv_mask is not None:
            inferred_myo_mask = self.geometric_reasoner.infer_myo_mask_from_lv_mask(
                lv_mask, thickness_ratio=0.15, bottom_exclude_ratio=0.25, use_random=True
            )
            if inferred_myo_mask is not None:
                inferred["myocardium"] = (inferred_myo_mask, self._mask_to_bbox(inferred_myo_mask, H, W))
        
        if lv_mask is not None and rv_mask is not None:
            inferred_septum_mask = self.geometric_reasoner.infer_septum_mask_from_lv_rv_masks(
                lv_mask, rv_mask, use_random=True
            )
            if inferred_septum_mask is not None:
                inferred["interventricular_septum"] = (inferred_septum_mask, self._mask_to_bbox(inferred_septum_mask, H, W))
        
        if la_mask is not None and ra_mask is not None:
            inferred_ias_mask = self.geometric_reasoner.infer_interatrial_septum_mask_from_la_ra_masks(
                la_mask, ra_mask, use_random=True
            )
            if inferred_ias_mask is not None:
                inferred["interatrial_septum"] = (inferred_ias_mask, self._mask_to_bbox(inferred_ias_mask, H, W))
        
        if lv_mask is not None and la_mask is not None:
            inferred_mv_mask = self.geometric_reasoner.infer_mitral_valve_mask_from_lv_la_masks(
                lv_mask, la_mask, use_random=True
            )
            if inferred_mv_mask is not None:
                inferred["mitral_valve"] = (inferred_mv_mask, self._mask_to_bbox(inferred_mv_mask, H, W))
        
        if rv_mask is not None and ra_mask is not None:
            inferred_tv_mask = self.geometric_reasoner.infer_tricuspid_valve_mask_from_rv_ra_masks(
                rv_mask, ra_mask, use_random=True
            )
            if inferred_tv_mask is not None:
                inferred["tricuspid_valve"] = (inferred_tv_mask, self._mask_to_bbox(inferred_tv_mask, H, W))
        
        return inferred

    def _compute_pseudo_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: Tuple,
        prompt_names: List[str],
        is_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Pseudo loss 计算"""
        device = targets["boxes"].device
        dtype = targets["boxes"].dtype
        zero_tensor = torch.tensor(0.0, device=device, dtype=dtype)
        
        result = {
            CORE_LOSS_KEY: zero_tensor.clone(),
            "loss_pseudo_mask": zero_tensor.clone(),
            "loss_pseudo_bbox": zero_tensor.clone(),
            "loss_pseudo_cls": zero_tensor.clone(),
        }
        
        if is_aux:
            return result
        
        if not self.use_pseudo_labels or self.geometric_reasoner is None:
            return result
        
        if "pred_masks" not in outputs or outputs["pred_masks"] is None:
            return result

        pred_masks = outputs["pred_masks"]
        if pred_masks.dim() == 5:
            pred_masks = pred_masks.squeeze(2)

        H, W = pred_masks.shape[2], pred_masks.shape[3]

        find_img_ids = outputs.get("find_img_ids", [])
        batch_idx_indices, _, tgt_idx_indices = indices
        unique_img_ids = torch.unique(find_img_ids).tolist()

        img_id_to_main_structures: Dict[int, Dict[str, Tuple[torch.Tensor, int]]] = {
            img_id: {} for img_id in unique_img_ids
        }
        for i_in_match in range(len(batch_idx_indices)):
            batch_idx = batch_idx_indices[i_in_match].item()
            tgt_idx = tgt_idx_indices[i_in_match].item() if tgt_idx_indices is not None else i_in_match
            if batch_idx >= len(find_img_ids) or batch_idx >= len(prompt_names):
                continue
            img_id = find_img_ids[batch_idx].item()
            prompt_name = prompt_names[batch_idx]
            if prompt_name in MAIN_STRUCTURES:
                img_id_to_main_structures[img_id][prompt_name] = (targets["boxes"][tgt_idx], tgt_idx)

        batch_indices_with_gt: Set[int] = {
            batch_idx_indices[i].item() for i in range(len(batch_idx_indices))
        }
        img_batch_indices_map: Dict[int, List[int]] = {}
        for i in range(len(find_img_ids)):
            iid = find_img_ids[i].item()
            img_batch_indices_map.setdefault(iid, []).append(i)

        gt_masks_raw = targets.get("masks", None)
        if gt_masks_raw is None:
            return result

        pred_logits = outputs.get("pred_logits")
        if pred_logits is not None:
            pred_logits = pred_logits.squeeze(-1)

        loss_mask_list = []
        loss_bbox_list = []
        loss_cls_pos_list = []
        loss_cls_neg_list = []

        for img_id in unique_img_ids:
            main_structures = img_id_to_main_structures[img_id]
            
            main_structures_xyxy: Dict[str, torch.Tensor] = {}
            for name, (box_cxcywh, _) in main_structures.items():
                cx, cy, w, h = box_cxcywh
                main_structures_xyxy[name] = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

            gt_masks: Dict[str, torch.Tensor] = {}
            for name, (_, tgt_idx) in main_structures.items():
                if tgt_idx >= gt_masks_raw.shape[0]:
                    continue
                gt_m = gt_masks_raw[tgt_idx]
                if gt_m.dim() == 3:
                    gt_m = gt_m.squeeze(0)
                if gt_m.shape[-2:] != (H, W):
                    gt_m = F.interpolate(gt_m.float().unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest").squeeze()
                gt_masks[name] = gt_m.bool()

            inferred_structures = self._infer_pseudo_structures(gt_masks, main_structures_xyxy, H, W, device)

            for batch_idx in img_batch_indices_map.get(img_id, []):
                if batch_idx >= len(prompt_names) or batch_idx in batch_indices_with_gt:
                    continue

                prompt_name = prompt_names[batch_idx]
                if prompt_name not in inferred_structures:
                    continue

                inferred_mask, inferred_bbox = inferred_structures[prompt_name]
                if inferred_mask is None or not inferred_mask.any() or inferred_bbox is None:
                    continue

                x1, y1, x2, y2 = inferred_bbox
                pseudo_box_cxcywh = torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])

                # 使用与推理一致的 query 选择方式：pred_logits * presence_score
                # 这与 _postprocess_out 和 postprocessors.py 的处理方式一致
                if pred_logits is not None and batch_idx < pred_logits.shape[0]:
                    logits_q = pred_logits[batch_idx].sigmoid()  # (Q,)
                    presence_logit_dec = outputs.get("presence_logit_dec")
                    if presence_logit_dec is not None and batch_idx < presence_logit_dec.shape[0]:
                        presence_score = presence_logit_dec[batch_idx].sigmoid()
                        if presence_score.dim() > 0:
                            presence_score = presence_score.squeeze()
                        combined_score = logits_q * presence_score
                    else:
                        combined_score = logits_q
                    best_query_idx = combined_score.argmax().item()
                else:
                    # Fallback: 使用 bbox cost
                    pred_boxes = outputs["pred_boxes"][batch_idx]
                    cost_bbox = torch.cdist(pred_boxes, pseudo_box_cxcywh.unsqueeze(0), p=1).squeeze(-1)
                    best_query_idx = cost_bbox.argmin().item()

                pred_mask = pred_masks[batch_idx, best_query_idx]
                pred_flat = pred_mask.flatten()
                target_flat = inferred_mask.float().flatten()
                intersection = (pred_flat.sigmoid() * target_flat).sum()
                union = pred_flat.sigmoid().sum() + target_flat.sum()
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                loss_bce = F.binary_cross_entropy_with_logits(pred_flat, target_flat, reduction="mean")
                loss_mask_list.append((1 - dice + loss_bce) * self.loss_mask_weight)

                pred_box = outputs["pred_boxes"][batch_idx, best_query_idx]
                loss_l1 = F.l1_loss(pred_box, pseudo_box_cxcywh, reduction="mean") * self.loss_bbox_weight
                pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_box.unsqueeze(0))
                loss_giou = (1 - box_ops.fast_diag_generalized_box_iou(pred_xyxy, inferred_bbox.unsqueeze(0))[0]) * self.loss_giou_weight
                loss_bbox_list.append(loss_l1 + loss_giou)

                if pred_logits is not None:
                    matched_logit = pred_logits[batch_idx, best_query_idx]
                    prob = matched_logit.sigmoid()
                    pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_box.unsqueeze(0))
                    iou = box_ops.fast_diag_box_iou(pred_xyxy, inferred_bbox.unsqueeze(0))[0]
                    soft_target = torch.clamp(prob.detach() ** 0.25 * iou ** 0.75, 0.01)
                    loss_pos = F.binary_cross_entropy_with_logits(matched_logit, soft_target, reduction="mean") * self.loss_cls_weight
                    loss_cls_pos_list.append(loss_pos)

                    num_queries = pred_logits.shape[1]
                    neg_indices = [q for q in range(num_queries) if q != best_query_idx]
                    if neg_indices:
                        neg_logits = pred_logits[batch_idx, neg_indices]
                        neg_prob = neg_logits.sigmoid()
                        focal_weight = neg_prob ** 2.0
                        loss_neg = (F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits), reduction="none") * focal_weight).mean() * self.loss_cls_weight
                        loss_cls_neg_list.append(loss_neg)

        if not loss_mask_list:
            return result

        loss_mask = torch.stack(loss_mask_list).mean()
        loss_bbox = torch.stack(loss_bbox_list).mean()
        loss_cls_pos = torch.stack(loss_cls_pos_list).mean() if loss_cls_pos_list else zero_tensor.clone()
        loss_cls_neg = torch.stack(loss_cls_neg_list).mean() if loss_cls_neg_list else zero_tensor.clone()

        total_loss = (loss_mask + loss_bbox + loss_cls_pos + loss_cls_neg) * self.pseudo_label_weight
        
        result[CORE_LOSS_KEY] = total_loss
        result["loss_pseudo_mask"] = loss_mask
        result["loss_pseudo_bbox"] = loss_bbox
        result["loss_pseudo_cls"] = loss_cls_pos + loss_cls_neg
        
        return result

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any],
        indices: Tuple,
        num_boxes: float,
        is_aux: bool = False,
        prompt_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward: 计算 few-shot loss 和 pseudo label loss"""
        device = targets["boxes"].device
        zero_tensor = torch.tensor(0.0, device=device)
        result = {CORE_LOSS_KEY: zero_tensor.clone()}
        
        if self.few_shot_loss is not None:
            few_shot_result = self.few_shot_loss(outputs, targets, indices, num_boxes, is_aux)
            result[CORE_LOSS_KEY] = result[CORE_LOSS_KEY] + few_shot_result.pop(CORE_LOSS_KEY, zero_tensor.clone())
            result.update(few_shot_result)
        
        if self.use_pseudo_labels:
            if prompt_names is None:
                prompt_names = outputs.get("find_prompt_names")
            if prompt_names is not None:
                pseudo_result = self._compute_pseudo_loss(outputs, targets, indices, prompt_names, is_aux)
                result[CORE_LOSS_KEY] = result[CORE_LOSS_KEY] + pseudo_result.pop(CORE_LOSS_KEY)
                result.update(pseudo_result)
        
        return result
