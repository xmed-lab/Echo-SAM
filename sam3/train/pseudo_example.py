#!/usr/bin/env python3
"""
Pseudo Mask 可视化脚本

用于可视化从 GT mask 推理出的 pseudo mask 效果：
1. 从 LV GT mask 推理 Myo mask
2. 从 Myo GT mask 推理 LV mask
3. 从 LV + RV GT mask 推理室间隔 mask

用法:
    python pseudo_example.py --img_folder <path> --json <path> --img_idx <index>
    python pseudo_example.py --img_folder CAMUS-A4C-ExMyo/5-shot --json CAMUS-A4C-ExMyo/5-shot/_annotations.coco.json --img_idx 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from sam3.model.geometric_reasoning import GeometricReasoner


COLORS = {
    "left_ventricle": (255, 0, 0),
    "right_ventricle": (0, 255, 0),
    "left_atrium": (0, 0, 255),
    "right_atrium": (255, 255, 0),
    "myocardium": (255, 0, 255),
    "interventricular_septum": (0, 255, 255),
    "interatrial_septum": (128, 128, 0),
    "mitral_valve": (128, 0, 128),
    "tricuspid_valve": (0, 128, 128),
    "inferred_myo": (255, 128, 128),
    "inferred_lv": (128, 255, 128),
    "inferred_septum": (128, 128, 255),
    "inferred_lv_bbox": (255, 165, 0),
    "inferred_lv_ellipse": (255, 200, 100),
    "inferred_interatrial_septum": (200, 200, 0),
    "inferred_mitral_valve": (200, 0, 200),
    "inferred_tricuspid_valve": (0, 200, 200),
}


def load_coco_annotations(json_path: str, img_folder: str):
    """加载 COCO 格式标注"""
    full_json_path = os.path.join(project_root, json_path)
    full_img_folder = os.path.join(project_root, img_folder)
    
    if not os.path.exists(full_json_path):
        raise FileNotFoundError(f"JSON file not found: {full_json_path}")
    
    coco = COCO(full_json_path)
    return coco, full_img_folder


def get_image_and_masks(coco: COCO, img_folder: str, img_idx: int):
    """获取指定索引的图像和 mask"""
    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {cat["id"]: cat["name"] for cat in cats}
    print(f"Categories in dataset: {id_to_name}")
    
    img_ids = coco.getImgIds()
    if img_idx >= len(img_ids):
        raise ValueError(f"img_idx {img_idx} out of range (max: {len(img_ids) - 1})")
    
    img_id = img_ids[img_idx]
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_folder, img_info["file_name"])
    
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img_info["height"], img_info["width"]
    
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    gt_masks = {}
    gt_bboxes = {}
    
    for ann in anns:
        cat_id = ann["category_id"]
        name = id_to_name.get(cat_id, f"unknown_{cat_id}")
        
        if "segmentation" in ann and ann["segmentation"]:
            mask = coco.annToMask(ann)
            gt_masks[name] = torch.from_numpy(mask).bool()
        
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            x1_norm = x / W
            y1_norm = y / H
            x2_norm = (x + w) / W
            y2_norm = (y + h) / H
            gt_bboxes[name] = torch.tensor([x1_norm, y1_norm, x2_norm, y2_norm])
    
    return img, gt_masks, gt_bboxes, img_info


def infer_pseudo_masks(
    gt_masks: dict,
    gt_bboxes: dict,
    geometric_reasoner: GeometricReasoner,
):
    """推理 pseudo masks"""
    H, W = None, None
    for mask in gt_masks.values():
        H, W = mask.shape
        break
    
    if H is None:
        return {}, {}
    
    inferred_masks = {}
    inferred_bboxes = {}
    
    has_lv = any("left_ventricle" in name.lower() or name.lower() == "lv" for name in gt_masks.keys())
    has_myo = any("myocardium" in name.lower() or name.lower() == "myo" for name in gt_masks.keys())
    
    print(f"Has LV: {has_lv}, Has Myo: {has_myo}")
    
    lv_mask = None
    for name, mask in gt_masks.items():
        if "left_ventricle" in name.lower() or name.lower() == "lv":
            lv_mask = mask
            break
    
    rv_mask = None
    for name, mask in gt_masks.items():
        if "right_ventricle" in name.lower() or name.lower() == "rv":
            rv_mask = mask
            break
    
    la_mask = None
    for name, mask in gt_masks.items():
        if "left_atrium" in name.lower() or name.lower() == "la":
            la_mask = mask
            break
    
    ra_mask = None
    for name, mask in gt_masks.items():
        if "right_atrium" in name.lower() or name.lower() == "ra":
            ra_mask = mask
            break
    
    if lv_mask is None and la_mask is not None and rv_mask is not None:
        print("Inferring LV from LA and RV (high confidence)...")
        rv_bbox = None
        ra_bbox = None
        la_bbox = None
        
        for name, bbox in gt_bboxes.items():
            if "right_ventricle" in name.lower() or name.lower() == "rv":
                rv_bbox = bbox
            elif "right_atrium" in name.lower() or name.lower() == "ra":
                ra_bbox = bbox
            elif "left_atrium" in name.lower() or name.lower() == "la":
                la_bbox = bbox
        
        if rv_bbox is not None or la_bbox is not None:
            lv_bbox = geometric_reasoner.infer_lv(
                rv_bbox=rv_bbox,
                ra_bbox=ra_bbox,
                la_bbox=la_bbox,
                image_width=1.0,
                use_random=False,
            )
            if lv_bbox is not None:
                inferred_bboxes["inferred_lv_bbox"] = lv_bbox
                print(f"  Inferred LV bbox: {lv_bbox.tolist()}")
                
                lv_ellipse_mask = geometric_reasoner.generate_ellipse_mask_from_bbox(
                    lv_bbox, H, W
                )
                if lv_ellipse_mask is not None:
                    lv_mask = lv_ellipse_mask
                    inferred_masks["inferred_lv_ellipse"] = lv_ellipse_mask
                    print(f"  Generated LV ellipse mask: {lv_ellipse_mask.sum().item()} pixels")
    
    if lv_mask is not None:
        print("Inferring Myo from LV...")
        myo_mask = geometric_reasoner.infer_myo_mask_from_lv_mask(
            lv_mask,
            thickness_ratio=0.15,
            bottom_exclude_ratio=0.25,
        )
        if myo_mask is not None:
            inferred_masks["inferred_myo"] = myo_mask
            ys, xs = torch.where(myo_mask)
            if len(ys) > 0:
                y1, y2 = ys.min().item(), ys.max().item()
                x1, x2 = xs.min().item(), xs.max().item()
                inferred_bboxes["inferred_myo"] = torch.tensor([x1, y1, x2, y2])
            print(f"  Inferred Myo: {myo_mask.sum().item()} pixels")
    
    if lv_mask is not None and rv_mask is not None:
        print("Inferring Septum from LV and RV masks...")
        septum_mask = geometric_reasoner.infer_septum_mask_from_lv_rv_masks(
            lv_mask, rv_mask
        )
        if septum_mask is not None:
            inferred_masks["inferred_septum"] = septum_mask
            ys, xs = torch.where(septum_mask)
            if len(ys) > 0:
                y1, y2 = ys.min().item(), ys.max().item()
                x1, x2 = xs.min().item(), xs.max().item()
                inferred_bboxes["inferred_septum"] = torch.tensor([x1, y1, x2, y2])
            print(f"  Inferred Septum: {septum_mask.sum().item()} pixels")
    
    if la_mask is not None and ra_mask is not None:
        print("Inferring Interatrial Septum from LA and RA masks...")
        ias_mask = geometric_reasoner.infer_interatrial_septum_mask_from_la_ra_masks(
            la_mask, ra_mask
        )
        if ias_mask is not None:
            inferred_masks["inferred_interatrial_septum"] = ias_mask
            ys, xs = torch.where(ias_mask)
            if len(ys) > 0:
                y1, y2 = ys.min().item(), ys.max().item()
                x1, x2 = xs.min().item(), xs.max().item()
                inferred_bboxes["inferred_interatrial_septum"] = torch.tensor([x1, y1, x2, y2])
            print(f"  Inferred Interatrial Septum: {ias_mask.sum().item()} pixels")
    
    if lv_mask is not None and la_mask is not None:
        print("Inferring Mitral Valve from LV and LA masks...")
        mv_mask = geometric_reasoner.infer_mitral_valve_mask_from_lv_la_masks(
            lv_mask, la_mask
        )
        if mv_mask is not None:
            inferred_masks["inferred_mitral_valve"] = mv_mask
            ys, xs = torch.where(mv_mask)
            if len(ys) > 0:
                y1, y2 = ys.min().item(), ys.max().item()
                x1, x2 = xs.min().item(), xs.max().item()
                inferred_bboxes["inferred_mitral_valve"] = torch.tensor([x1, y1, x2, y2])
            print(f"  Inferred Mitral Valve: {mv_mask.sum().item()} pixels")
    
    if rv_mask is not None and ra_mask is not None:
        print("Inferring Tricuspid Valve from RV and RA masks...")
        tv_mask = geometric_reasoner.infer_tricuspid_valve_mask_from_rv_ra_masks(
            rv_mask, ra_mask
        )
        if tv_mask is not None:
            inferred_masks["inferred_tricuspid_valve"] = tv_mask
            ys, xs = torch.where(tv_mask)
            if len(ys) > 0:
                y1, y2 = ys.min().item(), ys.max().item()
                x1, x2 = xs.min().item(), xs.max().item()
                inferred_bboxes["inferred_tricuspid_valve"] = torch.tensor([x1, y1, x2, y2])
            print(f"  Inferred Tricuspid Valve: {tv_mask.sum().item()} pixels")
    
    return inferred_masks, inferred_bboxes


def visualize_results(
    img: np.ndarray,
    gt_masks: dict,
    gt_bboxes: dict,
    inferred_masks: dict,
    inferred_bboxes: dict,
    save_path: str = None,
):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    overlay_gt = img.copy().astype(float)
    for name, mask in gt_masks.items():
        mask_np = mask.numpy()
        color = COLORS.get(name, (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)))
        color = np.array(color) / 255.0
        for c in range(3):
            overlay_gt[:, :, c] = np.where(
                mask_np,
                overlay_gt[:, :, c] * 0.5 + color[c] * 255 * 0.5,
                overlay_gt[:, :, c]
            )
    
    H, W = img.shape[:2]
    for name, bbox in gt_bboxes.items():
        x1, y1, x2, y2 = bbox.numpy()
        x1, x2 = int(x1 * W), int(x2 * W)
        y1, y2 = int(y1 * H), int(y2 * H)
        color = COLORS.get(name, (255, 255, 255))
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              fill=False, edgecolor=np.array(color)/255.0, linewidth=2)
        axes[1].add_patch(rect)
    
    axes[1].imshow(overlay_gt.astype(np.uint8))
    axes[1].set_title(f"GT Masks ({len(gt_masks)} masks)")
    axes[1].axis("off")
    
    overlay_inferred = img.copy().astype(float)
    for name, mask in inferred_masks.items():
        mask_np = mask.numpy()
        color = COLORS.get(name, (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255)))
        color = np.array(color) / 255.0
        for c in range(3):
            overlay_inferred[:, :, c] = np.where(
                mask_np,
                overlay_inferred[:, :, c] * 0.5 + color[c] * 255 * 0.5,
                overlay_inferred[:, :, c]
            )
    
    for name, bbox in inferred_bboxes.items():
        bbox_np = bbox.numpy()
        if name == "inferred_lv_bbox":
            x1, y1, x2, y2 = bbox_np
            x1, x2 = int(x1 * W), int(x2 * W)
            y1, y2 = int(y1 * H), int(y2 * H)
        else:
            x1, y1, x2, y2 = int(bbox_np[0]), int(bbox_np[1]), int(bbox_np[2]), int(bbox_np[3])
        color = COLORS.get(name, (255, 255, 255))
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              fill=False, edgecolor=np.array(color)/255.0, linewidth=2,
                              linestyle='--')
        axes[2].add_patch(rect)
    
    axes[2].imshow(overlay_inferred.astype(np.uint8))
    axes[2].set_title(f"Inferred Masks ({len(inferred_masks)} masks)")
    axes[2].axis("off")
    
    gt_legend = [f"{name}" for name in gt_masks.keys()]
    inferred_legend = [f"{name}" for name in inferred_masks.keys()]
    
    legend_text = "GT: " + ", ".join(gt_legend) + "\n"
    legend_text += "Inferred: " + ", ".join(inferred_legend)
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Pseudo Mask Visualization")
    parser.add_argument("--img_folder", type=str, required=True,
                        help="Path to image folder (relative to project root)")
    parser.add_argument("--json", type=str, required=True,
                        help="Path to COCO JSON file (relative to project root)")
    parser.add_argument("--img_idx", type=int, default=0,
                        help="Image index to visualize")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save visualization")
    parser.add_argument("--show", action="store_true",
                        help="Show visualization")
    args = parser.parse_args()
    
    print(f"Loading COCO annotations from {args.json}...")
    coco, img_folder = load_coco_annotations(args.json, args.img_folder)
    
    print(f"Loading image {args.img_idx}...")
    img, gt_masks, gt_bboxes, img_info = get_image_and_masks(coco, img_folder, args.img_idx)
    
    print(f"Image: {img_info['file_name']}")
    print(f"GT Masks: {list(gt_masks.keys())}")
    print(f"GT BBoxes: {list(gt_bboxes.keys())}")
    
    geometric_reasoner = GeometricReasoner()
    
    print("Inferring pseudo masks...")
    inferred_masks, inferred_bboxes = infer_pseudo_masks(
        gt_masks, gt_bboxes, geometric_reasoner
    )
    
    print(f"Inferred Masks: {list(inferred_masks.keys())}")
    print(f"Inferred BBoxes: {list(inferred_bboxes.keys())}")
    
    for name, mask in inferred_masks.items():
        area = mask.sum().item()
        print(f"  {name}: {area} pixels")
    
    save_path = args.save
    if save_path is None and not args.show:
        save_path = f"pseudo_vis_{args.img_idx}.png"
    
    visualize_results(
        img, gt_masks, gt_bboxes,
        inferred_masks, inferred_bboxes,
        save_path=save_path
    )


if __name__ == "__main__":
    main()
