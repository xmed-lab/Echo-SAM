import gc
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.model_misc import SAM3Output
from sam3.visualization_utils import plot_results_to_array, COLORS
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.utils.checkpoint_utils import load_state_dict_into_model
from sam3.train.utils.distributed import barrier, get_rank
from sam3.train.utils.logger import setup_logging
from sam3.train.utils.train_utils import (
    AverageMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    is_dist_avail_and_initialized,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


_DEFAULT_VIS_NORM_MEAN = [0.5, 0.5, 0.5]
_DEFAULT_VIS_NORM_STD = [0.5, 0.5, 0.5]

gt_prompt_names = ["myocardium", "left_ventricle"]
extra_prompt_names = ["interatrial_septum", "interventricular_septum", "mitral_valve", "tricuspid_valve"]

VIS_COLOR = [0, 100, 0]


def _denorm_img_for_vis(img_tensor, mean=None, std=None):
    if mean is None:
        mean = _DEFAULT_VIS_NORM_MEAN
    if std is None:
        std = _DEFAULT_VIS_NORM_STD
    mean = torch.tensor(mean, dtype=img_tensor.dtype, device=img_tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=img_tensor.dtype, device=img_tensor.device).view(-1, 1, 1)
    return (img_tensor * std + mean).clamp(0.0, 1.0)


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


class Evaluator:
    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        accelerator: str = "cuda",
        seed_value: int = 123,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        meters: Optional[Dict[str, Any]] = None,
        amp: Optional[Dict[str, Any]] = None,
        output_dir: str = None,
    ):
        self._setup_env_variables(env_variables)

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.meters_conf = meters
        self.amp_enabled = amp.get("enabled", False) if amp else False
        self.amp_dtype = amp.get("amp_dtype", "float16") if amp else "float16"
        self.output_dir = output_dir or logging.get("log_dir")

        self._setup_device(accelerator)
        self._setup_torch_dist_and_backend(cuda or {}, distributed or {})

        makedir(self.output_dir)
        setup_logging(
            __name__,
            output_dir=self.output_dir,
            rank=self.rank,
            log_level_primary=logging.get("log_level_primary", "INFO"),
            log_level_secondary=logging.get("log_level_secondary", "ERROR"),
        )

        set_seeds(seed_value, 1, self.distributed_rank)

        assert is_dist_avail_and_initialized(), (
            "Torch distributed needs to be initialized before calling the evaluator."
        )

        self._setup_components()
        self._move_to_device()
        self._setup_dataloaders()

        self.load_checkpoint()
        self._setup_ddp_distributed(distributed or {}, accelerator)
        barrier()

    def _setup_env_variables(self, env_variables_conf):
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = cuda_conf.get("cudnn_benchmark", True)
            torch.backends.cudnn.deterministic = cuda_conf.get("cudnn_deterministic", False)

        backend = distributed_conf.get("backend", "nccl" if self.device.type == "cuda" else "gloo")
        self.rank = setup_distributed_backend(backend, distributed_conf.get("timeout_mins", 30))

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.get("find_unused_parameters", False),
        )

    def _move_to_device(self):
        logging.info(f"Moving model to device {self.device}")
        self.model.to(self.device)

    def _setup_components(self):
        logging.info("Setting up model...")
        self.epoch = 0
        self.model = instantiate(self.model_conf, _convert_="all")

        self.meters = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        logging.info("Finished setting up model")

    def _setup_dataloaders(self):
        self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

    def load_checkpoint(self):
        ckpt_path = self.checkpoint_conf.get("resume_from")
        if ckpt_path is None:
            ckpt_path = os.path.join(self.checkpoint_conf.get("save_dir", ""), "checkpoint.pt")
        
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_checkpoint(ckpt_path)
        else:
            logging.warning(f"No checkpoint found at {ckpt_path}")

    def _load_checkpoint(self, ckpt_path: str):
        logging.info(f"Loading checkpoint from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.get("skip_saving_parameters", []),
        )

        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        
        logging.info(f"Loaded checkpoint from {ckpt_path}")

    def _find_meter(self, phase: str, key: str):
        if self.meters and phase in self.meters and key in self.meters[phase]:
            return self.meters[phase][key]
        return None

    def _step(self, batch: BatchedDatapoint, model: nn.Module):
        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        find_stages = model(batch)
        batch_size = len(batch.img_batch)

        vis_data = None
        if isinstance(find_stages, SAM3Output):
            vis_data = (find_stages, batch, key)

        meters_dict = self._find_meter(Phase.VAL, key)
        if meters_dict is not None:
            for _, meter in meters_dict.items():
                meter.update(
                    find_stages=find_stages,
                    find_metadatas=batch.find_metadatas,
                    model=model,
                    batch=batch,
                    key=key,
                )

        return batch_size, vis_data

    def run(self):
        self.run_val()

    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)
        log_freq = self.logging_conf.get("log_freq", 10)

        self.model.eval()
        if hasattr(unwrap_ddp_if_wrapped(self.model), "on_validation_epoch_start"):
            unwrap_ddp_if_wrapped(self.model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem],
            {},
            prefix=f"Eval Epoch: [{self.epoch}]",
        )

        end = time.time()
        vis_counter = 0

        for data_iter, batch in enumerate(val_loader):
            data_time.update(time.time() - end)

            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self.amp_enabled,
                    dtype=get_amp_type(self.amp_dtype) if self.amp_enabled else None,
                ):
                    batch_size, vis_data = self._step(batch, self.model)

            if vis_data is not None:
                n_saved = self._save_vis(vis_data, phase, vis_counter)
                vis_counter += n_saved

            batch_time.update(time.time() - end)
            end = time.time()

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % log_freq == 0:
                progress.display(data_iter)

            if data_iter % 10 == 0:
                torch.distributed.barrier()

        if hasattr(unwrap_ddp_if_wrapped(self.model), "on_validation_epoch_end"):
            unwrap_ddp_if_wrapped(self.model).on_validation_epoch_end()

        out_dict = self._log_meters([phase])
        logging.info(f"Eval Results: {out_dict}")
        return out_dict

    def _log_meters(self, phases):
        out_dict = {}
        for phase in phases:
            if phase not in self.meters:
                continue
            for key, key_meters in self.meters[phase].items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    if hasattr(meter, "compute"):
                        out_dict[f"{phase}_{key}/{name}"] = meter.compute()
        return out_dict

    def _save_vis(self, vis_data, phase: str, vis_counter: int = 0):
        if not vis_data or get_rank() != 0:
            return 0
        find_stages, batch, key = vis_data
        
        eval_results_dir = os.path.join(self.output_dir, "eval_results")
        makedir(eval_results_dir)
        
        for structure_name in gt_prompt_names:
            makedir(os.path.join(eval_results_dir, structure_name, "bbox"))
            makedir(os.path.join(eval_results_dir, structure_name, "mask"))
        for structure_name in extra_prompt_names:
            makedir(os.path.join(eval_results_dir, structure_name, "bbox"))
            makedir(os.path.join(eval_results_dir, structure_name, "mask"))
        
        n_saved = 0
        try:
            with SAM3Output.iteration_mode(
                find_stages, iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
            ) as stages:
                stage_list = list(stages)
            if not stage_list:
                return 0
            out = stage_list[0]
            pred_masks = out.get("pred_masks")
            if pred_masks is None:
                return 0
            imgs = batch.img_batch
            if imgs is None or imgs.numel() == 0:
                return 0
            B = imgs.shape[0]
            pred_masks = pred_masks.float().sigmoid()
            pred_boxes_xyxy = out.get("pred_boxes_xyxy")
            pred_logits = out.get("pred_logits")
            presence_logit_dec = out.get("presence_logit_dec")

            find_inputs = getattr(batch, "find_inputs", None) or []
            find_targets = getattr(batch, "find_targets", None)
            find_metadatas = getattr(batch, "find_metadatas", None)
            stage_target = find_targets[0] if find_targets else None
            stage_input = find_inputs[0] if find_inputs else None
            stage_metadata = find_metadatas[0] if find_metadatas else None
            
            if stage_input is not None and hasattr(stage_input, "img_ids"):
                img_ids = stage_input.img_ids
                text_ids = stage_input.text_ids
                text_batch = getattr(batch, "find_text_batch", None) or []
            else:
                img_ids = torch.arange(B, device=imgs.device, dtype=torch.long)
                text_ids = None
                text_batch = []

            if img_ids.dim() == 0:
                img_ids = img_ids.unsqueeze(0)
            uniq_img_ids = torch.unique_consecutive(torch.sort(img_ids.flatten())[0])

            import cv2
            for vis_idx, u in enumerate(uniq_img_ids.tolist()):
                inds = (img_ids.flatten() == u).nonzero(as_tuple=True)[0]
                inds = inds.cpu().tolist()
                if not inds:
                    continue
                
                structure_to_inds = {}
                for i in inds:
                    if text_batch and text_ids is not None and text_ids[i].item() < len(text_batch):
                        struct_name = text_batch[text_ids[i].item()]
                        if struct_name not in structure_to_inds:
                            structure_to_inds[struct_name] = []
                        structure_to_inds[struct_name].append(i)

                if not structure_to_inds:
                    continue

                i0 = inds[0]
                img = imgs[i0].float().cpu()
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                img = _denorm_img_for_vis(img)
                H, W = img.shape[-2], img.shape[-1]
                img_np = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

                file_name = f"{vis_counter + n_saved:05d}"
                if stage_metadata is not None:
                    if hasattr(stage_metadata, "coco_image_id") and i0 < len(stage_metadata.coco_image_id):
                        coco_id = stage_metadata.coco_image_id[i0]
                        if coco_id >= 0:
                            file_name = f"img_{coco_id:06d}"
                    elif hasattr(stage_metadata, "original_image_id") and i0 < len(stage_metadata.original_image_id):
                        orig_id = stage_metadata.original_image_id[i0]
                        if orig_id >= 0:
                            file_name = f"img_{orig_id:06d}"

                def _get_pred_data(idx, include_mask=True):
                    logits_q = pred_logits[idx].float().sigmoid() if pred_logits is not None else None
                    presence_score = presence_logit_dec[idx].float().sigmoid() if presence_logit_dec is not None else None
                    
                    best_slot = 0
                    if (
                        logits_q is not None
                        and logits_q.numel() > 1
                        and pred_masks.dim() == 4
                        and pred_masks.size(1) > 1
                    ):
                        logits_q_flat = logits_q.flatten()
                        presence_score_flat = presence_score.flatten()[0] if presence_score is not None and presence_score.numel() > 0 else 1.0
                        
                        if presence_score is not None:
                            out_probs = logits_q_flat * presence_score_flat
                        else:
                            out_probs = logits_q_flat
                        best_slot = int(out_probs.argmax().item())

                    mask = None
                    if include_mask:
                        mask = pred_masks[idx].cpu()
                        if mask.dim() == 3 and mask.size(0) > 1:
                            mask = mask[best_slot : best_slot + 1]
                        elif mask.dim() == 3:
                            mask = mask[0:1]
                        if mask.shape[-2:] != (H, W):
                            mask = torch.nn.functional.interpolate(
                                mask.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                            ).squeeze(0)
                    
                    box_pixel = None
                    conf = 0.0
                    if pred_boxes_xyxy is not None and pred_boxes_xyxy.dim() >= 2:
                        if pred_boxes_xyxy.dim() == 3 and pred_boxes_xyxy.size(1) > 1 and logits_q is not None and logits_q.numel() > 1:
                            box_norm = pred_boxes_xyxy[idx, best_slot].cpu().float().clamp(0.0, 1.0)
                            logits_q_flat = logits_q.flatten()
                            presence_score_flat = presence_score.flatten()[0] if presence_score is not None and presence_score.numel() > 0 else 1.0
                            if presence_score is not None:
                                conf = (logits_q_flat[best_slot] * presence_score_flat).item()
                            else:
                                conf = logits_q_flat[best_slot].item()
                        else:
                            box_norm = pred_boxes_xyxy[idx, 0] if pred_boxes_xyxy.dim() == 3 else pred_boxes_xyxy[idx]
                            box_norm = box_norm.cpu().float().clamp(0.0, 1.0)
                            logits_q_flat = logits_q.flatten() if logits_q is not None else None
                            presence_score_flat = presence_score.flatten()[0] if presence_score is not None and presence_score.numel() > 0 else 1.0
                            if logits_q_flat is not None and logits_q_flat.numel() > 0:
                                if presence_score is not None:
                                    conf = (logits_q_flat[0] * presence_score_flat).item()
                                else:
                                    conf = logits_q_flat[0].item()
                            else:
                                conf = 0.0
                        box_pixel = box_norm * torch.tensor([W, H, W, H], dtype=box_norm.dtype, device=box_norm.device)
                    
                    return mask, box_pixel, conf

                def _plot_mask_only(img_np, masks, labels):
                    result_img = img_np.copy()
                    for i, mask in enumerate(masks):
                        if isinstance(mask, torch.Tensor):
                            mask = mask.squeeze(0).cpu().numpy()
                        if mask.shape[-2:] != (img_np.shape[0], img_np.shape[1]):
                            mask = cv2.resize(
                                mask.astype(np.float32),
                                (img_np.shape[1], img_np.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        mask = (mask > 0.5).astype(np.uint8)
                        # if labels is not None and i < len(labels) and labels[i]:
                        #     color_idx = _label_to_color_index(labels[i], n_colors)
                        # else:
                        #     color_idx = i % n_colors
                        # color = COLORS[color_idx].tolist()
                        # color_bgr = [int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)]
                        color_bgr = [int(VIS_COLOR[2]), int(VIS_COLOR[1]), int(VIS_COLOR[0])]
                        
                        mask_colored = np.zeros_like(img_np)
                        mask_colored[mask > 0] = color_bgr
                        result_img = cv2.addWeighted(result_img, 1.0, mask_colored, 1.0, 0)
                    return result_img

                for structure_name, struct_inds in structure_to_inds.items():
                    is_gt_structure = structure_name in gt_prompt_names
                    is_extra_structure = structure_name in extra_prompt_names
                    
                    if not is_gt_structure and not is_extra_structure:
                        continue
                    
                    struct_labels = [structure_name] * len(struct_inds)
                    
                    list_masks = []
                    list_boxes = []
                    list_scores = []
                    for idx in struct_inds:
                        include_mask = is_gt_structure or is_extra_structure
                        mask, box_pixel, conf = _get_pred_data(idx, include_mask=include_mask)
                        if mask is not None:
                            list_masks.append(mask)
                        if box_pixel is not None:
                            list_boxes.append(box_pixel)
                            list_scores.append(conf)
                        else:
                            list_boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0]))
                            list_scores.append(0.0)
                    
                    if is_gt_structure:
                        results_bbox = {
                            "scores": torch.tensor(list_scores, dtype=torch.float32),
                            "masks": [torch.zeros(H, W, dtype=torch.float32)] * len(struct_inds),
                            "boxes": torch.stack(list_boxes, dim=0),
                        }
                        bbox_overlay_np = plot_results_to_array(img_np, results_bbox, labels=struct_labels, figsize=(10, 8), dpi=100)
                        bbox_path = os.path.join(eval_results_dir, structure_name, "bbox", f"{file_name}_pred.png")
                        cv2.imwrite(bbox_path, cv2.cvtColor(bbox_overlay_np, cv2.COLOR_RGB2BGR))
                        
                        if list_masks:
                            mask_overlay_np = _plot_mask_only(img_np, list_masks, struct_labels)
                            mask_path = os.path.join(eval_results_dir, structure_name, "mask", f"{file_name}_pred.png")
                            cv2.imwrite(mask_path, cv2.cvtColor(mask_overlay_np, cv2.COLOR_RGB2BGR))
                        
                        original_path = os.path.join(eval_results_dir, structure_name, f"{file_name}_original.png")
                        cv2.imwrite(original_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                        
                        if stage_target is not None and hasattr(stage_target, "boxes_padded"):
                            gt_masks = []
                            gt_boxes = []
                            gt_labels = []
                            seg_ptr = 0
                            num_queries = stage_target.num_boxes.shape[0]
                            for idx in range(num_queries):
                                n = stage_target.num_boxes[idx].item()
                                if idx in struct_inds and n > 0:
                                    for k in range(n):
                                        if getattr(stage_target, "segments", None) is not None and seg_ptr + k < stage_target.segments.shape[0]:
                                            seg = stage_target.segments[seg_ptr + k].cpu().float()
                                            if seg.dim() == 3:
                                                seg = seg[0]
                                            if seg.shape[-2:] != (H, W):
                                                seg = torch.nn.functional.interpolate(
                                                    seg.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest-exact"
                                                ).squeeze(0).squeeze(0)
                                            gt_masks.append(seg)
                                        else:
                                            gt_masks.append(torch.zeros(H, W, dtype=torch.float32))
                                        box_cxcywh = stage_target.boxes_padded[idx, k].cpu().float()
                                        cx, cy, w, h = box_cxcywh[0].item(), box_cxcywh[1].item(), box_cxcywh[2].item(), box_cxcywh[3].item()
                                        x0 = (cx - w * 0.5) * W
                                        y0 = (cy - h * 0.5) * H
                                        x1 = (cx + w * 0.5) * W
                                        y1 = (cy + h * 0.5) * H
                                        gt_boxes.append(torch.tensor([x0, y0, x1, y1], dtype=torch.float32))
                                        gt_labels.append(structure_name if n <= 1 else f"{structure_name}_{k}")
                                seg_ptr += n
                            if gt_masks:
                                results_gt_bbox = {
                                    "scores": torch.zeros(len(gt_masks), dtype=torch.float32),
                                    "masks": [torch.zeros(H, W, dtype=torch.float32)] * len(gt_masks),
                                    "boxes": torch.stack(gt_boxes, dim=0),
                                }
                                gt_bbox_np = plot_results_to_array(img_np, results_gt_bbox, labels=gt_labels, figsize=(10, 8), dpi=100)
                                gt_bbox_path = os.path.join(eval_results_dir, structure_name, "bbox", f"{file_name}_gt.png")
                                cv2.imwrite(gt_bbox_path, cv2.cvtColor(gt_bbox_np, cv2.COLOR_RGB2BGR))
                                
                                gt_mask_np = _plot_mask_only(img_np, gt_masks, gt_labels)
                                gt_mask_path = os.path.join(eval_results_dir, structure_name, "mask", f"{file_name}_gt.png")
                                cv2.imwrite(gt_mask_path, cv2.cvtColor(gt_mask_np, cv2.COLOR_RGB2BGR))
                    
                    elif is_extra_structure:
                        results_bbox = {
                            "scores": torch.tensor(list_scores, dtype=torch.float32),
                            "masks": [torch.zeros(H, W, dtype=torch.float32)] * len(struct_inds),
                            "boxes": torch.stack(list_boxes, dim=0),
                        }
                        bbox_overlay_np = plot_results_to_array(img_np, results_bbox, labels=struct_labels, figsize=(10, 8), dpi=100)
                        bbox_path = os.path.join(eval_results_dir, structure_name, "bbox", f"{file_name}_pred.png")
                        cv2.imwrite(bbox_path, cv2.cvtColor(bbox_overlay_np, cv2.COLOR_RGB2BGR))
                        
                        if list_masks:
                            mask_overlay_np = _plot_mask_only(img_np, list_masks, struct_labels)
                            mask_path = os.path.join(eval_results_dir, structure_name, "mask", f"{file_name}_pred.png")
                            cv2.imwrite(mask_path, cv2.cvtColor(mask_overlay_np, cv2.COLOR_RGB2BGR))
                        
                        original_path = os.path.join(eval_results_dir, structure_name, f"{file_name}_original.png")
                        cv2.imwrite(original_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                
                n_saved += 1
                if n_saved % 10 == 0:
                    logging.info(f"Saved {n_saved} visualizations")

        except Exception:
            logging.exception("_save_vis failed")
        finally:
            if isinstance(find_stages, SAM3Output):
                try:
                    with SAM3Output.iteration_mode(
                        find_stages, iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
                    ) as stages:
                        for fs in stages:
                            if isinstance(fs, dict):
                                for k in list(fs.keys()):
                                    del fs[k]
                except Exception:
                    pass
        
        return n_saved
