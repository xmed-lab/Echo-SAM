# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
This evaluator is meant for regular COCO mAP evaluation, for example on the COCO val set.

For Category mAP, we need the model to make predictions for all the categories on every single image.
In general, since the number of classes can be big, and the API model makes predictions individually for each pair (image, class),
we may need to split the inference process for a given image in several chunks.
"""

import logging
from collections import defaultdict

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from sam3.train.utils.distributed import is_main_process

try:
    from tidecv import datasets, TIDE

    HAS_TIDE = True
except ImportError:
    HAS_TIDE = False
    print("WARNING: TIDE not installed. Detailed analysis will not be available.")


# the COCO detection metrics (https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L460-L471)
COCO_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _rle_to_mask(rle, coco_obj=None, ann=None):
    """
    将 RLE 编码的 mask 转换为 numpy 数组 (bool 类型)
    """
    if isinstance(rle, dict) and 'size' in rle and 'counts' in rle:
        return mask_utils.decode(rle).astype(bool)
    if coco_obj is not None and ann is not None:
        try:
            rle_decoded = coco_obj.annToRLE(ann)
            return mask_utils.decode(rle_decoded).astype(bool)
        except Exception:
            return np.zeros(rle['size'], dtype=bool) if isinstance(rle, dict) and 'size' in rle else np.zeros((1, 1), dtype=bool)
    return np.zeros((1, 1), dtype=bool)


def _iou_from_masks(mask_gt, mask_dt):
    """
    从两个 numpy mask 计算 IoU
    """
    if not np.any(mask_gt) and not np.any(mask_dt):
        return 1.0
    intersection = np.logical_and(mask_gt, mask_dt)
    union = np.logical_or(mask_gt, mask_dt)
    if not np.any(union):
        return 0.0
    return float(np.sum(intersection) / np.sum(union))


def _dice_from_rle(rle_gt, rle_dt):
    """
    从两个 RLE mask 计算 DICE coefficient：DICE = 2*|A∩B|/(|A|+|B|)。
    利用 IoU 与 DICE 的关系：DICE = 2*IoU/(1+IoU)。
    """
    a_gt = mask_utils.area(rle_gt)
    a_dt = mask_utils.area(rle_dt)
    if a_gt == 0 and a_dt == 0:
        return 1.0
    if a_gt == 0 or a_dt == 0:
        return 0.0
    iou_arr = np.asarray(mask_utils.iou([rle_dt], [rle_gt], [0]))
    iou = float(iou_arr.ravel()[0]) if iou_arr.size > 0 else 0.0
    if np.isnan(iou) or iou <= 0:
        return 0.0
    return float(2.0 * iou / (1.0 + iou))


def _get_surface_points(mask):
    """
    获取 mask 的表面轮廓点
    """
    from scipy.ndimage import binary_erosion
    mask = mask.astype(bool)
    if not np.any(mask):
        return np.array([]).reshape(0, 2)
    eroded = binary_erosion(mask)
    contour = np.logical_and(mask, np.logical_not(eroded))
    y, x = np.where(contour)
    return np.column_stack((y, x))


def _directed_hausdorff_distance(surface_a, surface_b, percentile=95):
    """
    计算从 surface_a 到 surface_b 的 directed Hausdorff distance
    """
    if len(surface_a) == 0 or len(surface_b) == 0:
        return float('inf')
    dists = np.sqrt(np.sum((surface_a[:, None, :] - surface_b[None, :, :]) ** 2, axis=2))
    min_dists = np.min(dists, axis=1)
    return float(np.percentile(min_dists, percentile))


def _hausdorff_distance_95(mask_gt, mask_dt):
    """
    计算 HD95 (95th percentile Hausdorff distance)
    """
    surface_gt = _get_surface_points(mask_gt)
    surface_dt = _get_surface_points(mask_dt)
    if len(surface_gt) == 0 and len(surface_dt) == 0:
        return 0.0
    if len(surface_gt) == 0 or len(surface_dt) == 0:
        return float('inf')
    hd_gt_to_dt = _directed_hausdorff_distance(surface_gt, surface_dt, 95)
    hd_dt_to_gt = _directed_hausdorff_distance(surface_dt, surface_gt, 95)
    return max(hd_gt_to_dt, hd_dt_to_gt)


def _average_symmetric_surface_distance(mask_gt, mask_dt):
    """
    计算 ASSD (Average Symmetric Surface Distance)
    """
    surface_gt = _get_surface_points(mask_gt)
    surface_dt = _get_surface_points(mask_dt)
    if len(surface_gt) == 0 and len(surface_dt) == 0:
        return 0.0
    if len(surface_gt) == 0 or len(surface_dt) == 0:
        return float('inf')
    
    dists_gt_to_dt = np.sqrt(np.sum((surface_gt[:, None, :] - surface_dt[None, :, :]) ** 2, axis=2))
    min_dists_gt_to_dt = np.min(dists_gt_to_dt, axis=1)
    
    dists_dt_to_gt = np.sqrt(np.sum((surface_dt[:, None, :] - surface_gt[None, :, :]) ** 2, axis=2))
    min_dists_dt_to_gt = np.min(dists_dt_to_gt, axis=1)
    
    return float((np.mean(min_dists_gt_to_dt) + np.mean(min_dists_dt_to_gt)) / 2.0)


def compute_coco_segmentation_metrics(cocoGt, cocoDt, positive_split=False):
    """
    在 COCO 分割结果上计算 mean DICE, mIOU, HD95, ASSD：
    对每个 GT 标注，按 (image_id, category_id) 匹配与之 IoU 最大的预测，
    再对该对计算各项指标，最后对所有 GT 取平均，同时返回每个 class 的单独统计。

    返回值中：
    - 'DICE_mean', 'mIOU', 'HD95', 'ASSD'：所有 class 的平均值
    - 'per_category'：dict，key 为 category_name，value 为该 class 的各项指标
    """
    imgIds = cocoGt.getImgIds()
    catIds = cocoGt.getCatIds()
    gt_anns = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=imgIds, catIds=catIds))
    dt_anns = cocoDt.loadAnns(cocoDt.getAnnIds(imgIds=imgIds, catIds=catIds))

    # cat_id -> category_name 映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in cocoGt.loadCats(catIds)}

    # 按 (image_id, category_id) 分组
    gt_by_ic = defaultdict(list)
    dt_by_ic = defaultdict(list)
    for ann in gt_anns:
        if ann.get("segmentation") is None:
            continue
        gt_by_ic[ann["image_id"], ann["category_id"]].append(ann)
    for ann in dt_anns:
        if ann.get("segmentation") is None:
            continue
        if positive_split:
            if (ann["image_id"], ann["category_id"]) not in gt_by_ic:
                continue
        dt_by_ic[ann["image_id"], ann["category_id"]].append(ann)

    # 全局列表
    dice_list = []
    iou_list = []
    hd95_list = []
    assd_list = []

    # 按 cat_id 分组的列表
    per_cat_dice  = defaultdict(list)
    per_cat_iou   = defaultdict(list)
    per_cat_hd95  = defaultdict(list)
    per_cat_assd  = defaultdict(list)

    for (im_id, cat_id), gts in gt_by_ic.items():
        dts = dt_by_ic.get((im_id, cat_id), [])
        if not dts:
            for _ in gts:
                dice_list.append(0.0);  per_cat_dice[cat_id].append(0.0)
                iou_list.append(0.0);   per_cat_iou[cat_id].append(0.0)
                hd95_list.append(float('inf')); per_cat_hd95[cat_id].append(float('inf'))
                assd_list.append(float('inf')); per_cat_assd[cat_id].append(float('inf'))
            continue

        for gt_ann in gts:
            try:
                rle_gt = cocoGt.annToRLE(gt_ann)
                mask_gt = _rle_to_mask(rle_gt)
            except Exception:
                dice_list.append(0.0);  per_cat_dice[cat_id].append(0.0)
                iou_list.append(0.0);   per_cat_iou[cat_id].append(0.0)
                hd95_list.append(float('inf')); per_cat_hd95[cat_id].append(float('inf'))
                assd_list.append(float('inf')); per_cat_assd[cat_id].append(float('inf'))
                continue

            best_iou = -1.0
            best_mask_dt = None
            best_dt_rle = None

            for dt_ann in dts:
                try:
                    rle_dt = cocoDt.annToRLE(dt_ann)
                    mask_dt = _rle_to_mask(rle_dt)
                except Exception:
                    continue
                current_iou = _iou_from_masks(mask_gt, mask_dt)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_mask_dt = mask_dt
                    best_dt_rle = rle_dt

            if best_mask_dt is None:
                dice_list.append(0.0);  per_cat_dice[cat_id].append(0.0)
                iou_list.append(0.0);   per_cat_iou[cat_id].append(0.0)
                hd95_list.append(float('inf')); per_cat_hd95[cat_id].append(float('inf'))
                assd_list.append(float('inf')); per_cat_assd[cat_id].append(float('inf'))
            else:
                dice = _dice_from_rle(rle_gt, best_dt_rle)
                hd95 = _hausdorff_distance_95(mask_gt, best_mask_dt)
                assd = _average_symmetric_surface_distance(mask_gt, best_mask_dt)

                dice_list.append(dice);  per_cat_dice[cat_id].append(dice)
                iou_list.append(best_iou); per_cat_iou[cat_id].append(best_iou)
                hd95_list.append(hd95);  per_cat_hd95[cat_id].append(hd95)
                assd_list.append(assd);  per_cat_assd[cat_id].append(assd)

    # 全局平均
    metrics = {}
    metrics['DICE_mean'] = float(np.mean(dice_list)) if dice_list else 0.0
    metrics['DICE_per_gt_count'] = len(dice_list)
    metrics['mIOU'] = float(np.mean(iou_list)) if iou_list else 0.0

    hd95_valid = [x for x in hd95_list if x != float('inf')]
    metrics['HD95'] = float(np.mean(hd95_valid)) if hd95_valid else float('inf')

    assd_valid = [x for x in assd_list if x != float('inf')]
    metrics['ASSD'] = float(np.mean(assd_valid)) if assd_valid else float('inf')

    # 每个 class 单独统计
    per_category = {}
    for cat_id in set(list(per_cat_dice.keys())):
        cat_name = cat_id_to_name.get(cat_id, str(cat_id))
        d_list = per_cat_dice[cat_id]
        i_list = per_cat_iou[cat_id]
        h_list = [x for x in per_cat_hd95[cat_id] if x != float('inf')]
        a_list = [x for x in per_cat_assd[cat_id] if x != float('inf')]
        per_category[cat_name] = {
            'DICE': float(np.mean(d_list)) if d_list else 0.0,
            'IOU':  float(np.mean(i_list)) if i_list else 0.0,
            'HD95': float(np.mean(h_list)) if h_list else float('inf'),
            'ASSD': float(np.mean(a_list)) if a_list else float('inf'),
            'count': len(d_list),
        }
    metrics['per_category'] = per_category

    return metrics


def compute_coco_segmentation_dice(cocoGt, cocoDt, positive_split=False):
    """
    旧函数，保持向后兼容
    """
    metrics = compute_coco_segmentation_metrics(cocoGt, cocoDt, positive_split)
    return metrics['DICE_mean'], {'dice_per_gt_count': metrics['DICE_per_gt_count']}


def convert_to_xywh(boxes):
    """Convert bounding boxes from xyxy format to xywh format."""
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


class HeapElement:
    """Utility class to make a heap with a custom comparator"""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class COCOevalCustom(COCOeval):
    """
    This is a slightly modified version of the original COCO API with added support for positive split evaluation.
    """

    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="segm", dt_only_positive=False
    ):
        super().__init__(cocoGt, cocoDt, iouType)
        self.dt_only_positive = dt_only_positive

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        _gts_cat_ids = defaultdict(set)  # gt for evaluation on positive split
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
            _gts_cat_ids[gt["image_id"]].add(gt["category_id"])

        #### BEGIN MODIFICATION ####
        for dt in dts:
            if (
                self.dt_only_positive
                and dt["category_id"] not in _gts_cat_ids[dt["image_id"]]
            ):
                continue
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        #### END MODIFICATION ####
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results


class CocoEvaluatorOfflineWithPredFileEvaluators:
    def __init__(
        self,
        gt_path,
        tide: bool = True,
        iou_type: str = "bbox",
        positive_split=False,
    ):
        self.gt_path = gt_path
        self.tide_enabled = HAS_TIDE and tide
        self.positive_split = positive_split
        self.iou_type = iou_type

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("OfflineCoco evaluator: Loading groundtruth")
        self.gt = COCO(self.gt_path)

        # Creating the result file
        logging.info("Coco evaluator: Creating the result file")
        cocoDt = self.gt.loadRes(str(dumped_file))

        # Run the evaluation
        logging.info("Coco evaluator: Running evaluation")
        coco_eval = COCOevalCustom(
            self.gt, cocoDt, iouType=self.iou_type, dt_only_positive=self.positive_split
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        outs = {}
        for i, value in enumerate(coco_eval.stats):
            outs[f"coco_eval_{self.iou_type}_{COCO_METRICS[i]}"] = value

        # 分割任务下额外计算 mean DICE, mIOU, HD95, ASSD
        # 说明：各项指标对每个 GT 取「同 (image_id,category_id) 下与某预测的最大 IoU」匹配后计算，
        # 故偏乐观；AP/AR 使用 IoU 阈值（如 0.5/0.75）做 TP/FP 匹配。DICE≈0.5 约对应 IoU≈0.33，
        # 因此 DICE 中等而 AP 较低属正常（零样本常为粗略分割，难以稳定达到 IoU>0.5）。
        if self.iou_type == "segm":
            metrics = compute_coco_segmentation_metrics(
                self.gt, cocoDt, positive_split=self.positive_split
            )
            # 全局平均指标
            outs["coco_eval_segm_DICE_mean"] = metrics['DICE_mean']
            outs["coco_eval_segm_mIOU"] = metrics['mIOU']
            outs["coco_eval_segm_HD95"] = metrics['HD95']
            outs["coco_eval_segm_ASSD"] = metrics['ASSD']
            logging.info(
                "Coco evaluator: DICE=%.4f, mIOU=%.4f, HD95=%.4f, ASSD=%.4f (over %d GT anns)",
                metrics['DICE_mean'], metrics['mIOU'],
                metrics['HD95'], metrics['ASSD'],
                metrics['DICE_per_gt_count'],
            )
            # 每个 class 单独指标
            per_category = metrics.get('per_category', {})
            if per_category:
                logging.info("Coco evaluator: Per-class segmentation metrics:")
                for cat_name, cat_metrics in sorted(per_category.items()):
                    logging.info(
                        "  %-30s  DICE=%.4f  IOU=%.4f  HD95=%.4f  ASSD=%.4f  (n=%d)",
                        cat_name,
                        cat_metrics['DICE'], cat_metrics['IOU'],
                        cat_metrics['HD95'], cat_metrics['ASSD'],
                        cat_metrics['count'],
                    )
                    safe_name = cat_name.replace(" ", "_")
                    outs[f"coco_eval_segm_DICE_{safe_name}"] = cat_metrics['DICE']
                    outs[f"coco_eval_segm_IOU_{safe_name}"]  = cat_metrics['IOU']
                    outs[f"coco_eval_segm_HD95_{safe_name}"] = cat_metrics['HD95']
                    outs[f"coco_eval_segm_ASSD_{safe_name}"] = cat_metrics['ASSD']

        if self.tide_enabled:
            logging.info("Coco evaluator: Loading TIDE")
            self.tide_gt = datasets.COCO(self.gt_path)
            self.tide = TIDE(mode="mask" if self.iou_type == "segm" else "bbox")

            # Run TIDE
            logging.info("Coco evaluator: Running TIDE")
            self.tide.evaluate(
                self.tide_gt, datasets.COCOResult(str(dumped_file)), name="coco_eval"
            )
            self.tide.summarize()
            for k, v in self.tide.get_main_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

            for k, v in self.tide.get_special_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

        return outs
