
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class GeometricReasoner:
    """
    心脏结构几何推理器：
    根据四个主要结构（LV, LA, RV, RA）的 bbox，推理出其他结构的位置
    
    BBox 格式：假设是 (x1, y1, x2, y2) - 左上角和右下角坐标
    """

    def __init__(self):
        pass

    def _get_bbox_center(self, bbox: torch.Tensor) -> torch.Tensor:
        """计算 bbox 的中心点"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return torch.stack([cx, cy])

    def _get_bbox_size(self, bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 bbox 的宽和高"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return w, h

    def infer_lv_from_rv_ra_la(self, rv_bbox: torch.Tensor, ra_bbox: torch.Tensor,
                                la_bbox: torch.Tensor,
                                use_random: bool = True, random_range: float = 0.1) -> torch.Tensor:
        """
        通过 RV、RA、LA 三个结构推理 LV（最优方案）

        解析：利用心脏对称性，RA→RV 的位移向量与 LA→LV 的位移向量相同：
            LV_center = LA_center + (RV_center - RA_center)

        该方法对 A4C 等任意旋转方向的心脏都成立，无需假设正交网格布局。
        大小参照 RV（左右心室尺寸相近）。

        Args:
            rv_bbox: 右心室 bbox
            ra_bbox: 右心房 bbox
            la_bbox: 左心房 bbox
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±5% 的大小缩放）

        Returns:
            推理出的左心室 bbox
        """
        rv_cx, rv_cy = self._get_bbox_center(rv_bbox)
        ra_cx, ra_cy = self._get_bbox_center(ra_bbox)
        la_cx, la_cy = self._get_bbox_center(la_bbox)
        rv_w, rv_h = self._get_bbox_size(rv_bbox)

        actual_scale = 1.0
        if use_random:
            actual_scale = 1.0 + (torch.rand(1).item() - 0.5) * 2 * random_range
            actual_scale = max(0.85, min(1.15, actual_scale))

        # RA→RV 位移向量即为 LA→LV 位移向量
        dx = rv_cx - ra_cx
        dy = rv_cy - ra_cy
        lv_cx = la_cx + dx
        lv_cy = la_cy + dy
        lv_w = rv_w * actual_scale
        lv_h = rv_h * actual_scale

        return torch.stack([lv_cx - lv_w / 2, lv_cy - lv_h / 2,
                            lv_cx + lv_w / 2, lv_cy + lv_h / 2])

    def infer_lv_from_rv_la(self, rv_bbox: torch.Tensor, la_bbox: torch.Tensor,
                             use_random: bool = True, random_range: float = 0.1) -> torch.Tensor:
        """
        通过 RV 和 LA 推理 LV（仅两个结构可用时的退化方案）

        解析：在标准正交4腔视图中，LA 和 LV 同列（x对齐），RV 和 LV 同行（y对齐）。
        注意：若心脏有旋转（如 A4C 切面），此方案可能偏差较大；
        有 RA 时应优先使用 infer_lv_from_rv_ra_la（向量位移法）。

        Args:
            rv_bbox: 右心室 bbox
            la_bbox: 左心房 bbox
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±5% 的大小缩放）

        Returns:
            推理出的左心室 bbox
        """
        rv_cx, rv_cy = self._get_bbox_center(rv_bbox)
        la_cx, _ = self._get_bbox_center(la_bbox)
        rv_w, rv_h = self._get_bbox_size(rv_bbox)

        actual_scale = 1.0
        if use_random:
            actual_scale = 1.0 + (torch.rand(1).item() - 0.5) * 2 * random_range
            actual_scale = max(0.85, min(1.15, actual_scale))

        lv_cx = la_cx
        lv_cy = rv_cy
        lv_w = rv_w * actual_scale
        lv_h = rv_h * actual_scale

        return torch.stack([lv_cx - lv_w / 2, lv_cy - lv_h / 2,
                            lv_cx + lv_w / 2, lv_cy + lv_h / 2])

    def infer_lv_from_rv_ra(self, rv_bbox: torch.Tensor, ra_bbox: torch.Tensor,
                             image_width: Optional[float] = None,
                             use_random: bool = True, random_range: float = 0.1) -> torch.Tensor:
        """
        通过 RV 和 RA 推理 LV（对称镜像方案）

        解析：利用心脏左右对称性，以室间隔为轴将 RV 镜像到对侧得到 LV。
        由于输入图像可能被水平翻转，需先判断 RV/RA 在图像的哪一侧：
        - RV/RA 在右侧（标准方向）：室间隔 ≈ min(rv_x1, ra_x1)，LV 在室间隔左侧
        - RV/RA 在左侧（图像翻转）：室间隔 ≈ max(rv_x2, ra_x2)，LV 在室间隔右侧

        侧别判断：
        - 若提供 image_width：比较 RV/RA 平均中心与图像中心
        - 若未提供 image_width：以 RV/RA bbox 的联合范围中心作为参考，
          即若 RV 中心偏向联合范围的右半部分，则认为 RV 在图像右侧

        Args:
            rv_bbox: 右心室 bbox
            ra_bbox: 右心房 bbox
            image_width: 图像宽度（像素或归一化坐标下的1.0），用于判断左右方向；
                         若为 None 则自动推断
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±5% 的大小缩放）

        Returns:
            推理出的左心室 bbox
        """
        rv_cx, rv_cy = self._get_bbox_center(rv_bbox)
        ra_cx, _ = self._get_bbox_center(ra_bbox)
        rv_w, rv_h = self._get_bbox_size(rv_bbox)

        # 判断 RV/RA 在图像左侧还是右侧
        rv_ra_cx = (rv_cx + ra_cx) / 2  # RV 和 RA 的平均中心 x
        if image_width is not None:
            # 有图像宽度：直接与图像中心比较
            rv_on_right = rv_ra_cx > image_width / 2
        else:
            # 无图像宽度：以 RV/RA 四条边的联合范围中点作为参考
            # 若 RV/RA 偏向联合范围右半部分，说明其在图像右侧
            union_left = torch.min(rv_bbox[0], ra_bbox[0])
            union_right = torch.max(rv_bbox[2], ra_bbox[2])
            union_cx = (union_left + union_right) / 2
            rv_on_right = rv_ra_cx > union_cx

        actual_scale = 1.0
        if use_random:
            actual_scale = 1.0 + (torch.rand(1).item() - 0.5) * 2 * random_range
            actual_scale = max(0.85, min(1.15, actual_scale))

        lv_w = rv_w * actual_scale
        lv_h = rv_h * actual_scale

        if rv_on_right:
            # 标准方向：RV/RA 在右，LV 在室间隔左侧
            septum_x = torch.min(rv_bbox[0], ra_bbox[0])
            lv_cx = septum_x - lv_w / 2
        else:
            # 翻转方向：RV/RA 在左，LV 在室间隔右侧
            septum_x = torch.max(rv_bbox[2], ra_bbox[2])
            lv_cx = septum_x + lv_w / 2

        lv_cy = rv_cy

        return torch.stack([lv_cx - lv_w / 2, lv_cy - lv_h / 2,
                            lv_cx + lv_w / 2, lv_cy + lv_h / 2])

    def infer_lv_from_ra_la(self, ra_bbox: torch.Tensor, la_bbox: torch.Tensor,
                             ventricle_to_atrium_ratio: float = 1.2,
                             use_random: bool = True, random_range: float = 0.05) -> torch.Tensor:
        """
        通过 RA 和 LA 推理 LV（心房推算方案）

        解析：LA 是 LV 的直接上方结构（二尖瓣相连）
        - x 位置：与 LA 中心对齐
        - y 位置：紧接 LA 底部向下，LA 底部即为二尖瓣位置
        - 大小：心室通常比心房稍大，用 LA+RA 平均值乘以比例估计

        Args:
            ra_bbox: 右心房 bbox
            la_bbox: 左心房 bbox
            ventricle_to_atrium_ratio: 心室与心房的大小比例（默认 1.2）
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±5% 的比例缩放）

        Returns:
            推理出的左心室 bbox
        """
        la_cx, _ = self._get_bbox_center(la_bbox)
        la_w, la_h = self._get_bbox_size(la_bbox)
        ra_w, ra_h = self._get_bbox_size(ra_bbox)

        actual_ratio = ventricle_to_atrium_ratio
        if use_random:
            actual_ratio = ventricle_to_atrium_ratio + (torch.rand(1).item() - 0.5) * 2 * random_range
            actual_ratio = max(0.9, min(1.5, actual_ratio))

        # 用 LA 和 RA 的平均大小估计心房尺寸，再乘以比例得到心室大小
        avg_atrium_w = (la_w + ra_w) / 2
        avg_atrium_h = (la_h + ra_h) / 2
        lv_w = avg_atrium_w * actual_ratio
        lv_h = avg_atrium_h * actual_ratio

        # LV 紧接在 LA 下方
        la_bottom = la_bbox[3]
        lv_cx = la_cx
        lv_cy = la_bottom + lv_h / 2

        return torch.stack([lv_cx - lv_w / 2, lv_cy - lv_h / 2,
                            lv_cx + lv_w / 2, lv_cy + lv_h / 2])

    def infer_lv(self,
                 rv_bbox: Optional[torch.Tensor] = None,
                 ra_bbox: Optional[torch.Tensor] = None,
                 la_bbox: Optional[torch.Tensor] = None,
                 image_width: Optional[float] = None,
                 use_random: bool = True) -> Optional[torch.Tensor]:
        """
        从已知的 RV、RA、LA 中推理 LV 位置（任意两个结构可推理）

        优先级：
          1. RV+RA+LA（最优）：向量位移法，LV = LA + (RV - RA)，适用于任意旋转方向
          2. RV+LA          ：正交网格法，仅在心脏接近正交排列时准确（退化方案）
          3. RV+RA          ：室间隔镜像法，需判断左右方向
          4. RA+LA          ：LA 下方估计，大小从心房推算

        Args:
            rv_bbox: 右心室 bbox（可选）
            ra_bbox: 右心房 bbox（可选）
            la_bbox: 左心房 bbox（可选）
            image_width: 图像宽度，用于 RV+RA 方案判断图像是否翻转；
                         若为 None 则自动推断
            use_random: 是否添加随机扰动

        Returns:
            推理出的左心室 bbox，若可用结构不足两个则返回 None
        """
        # 三者都有：向量位移法（最优，对任意心脏旋转方向成立）
        if rv_bbox is not None and ra_bbox is not None and la_bbox is not None:
            return self.infer_lv_from_rv_ra_la(rv_bbox, ra_bbox, la_bbox, use_random=use_random)

        # 仅 RV+LA：退化的正交网格法
        if rv_bbox is not None and la_bbox is not None:
            return self.infer_lv_from_rv_la(rv_bbox, la_bbox, use_random=use_random)

        # 仅 RV+RA：室间隔镜像法
        if rv_bbox is not None and ra_bbox is not None:
            return self.infer_lv_from_rv_ra(rv_bbox, ra_bbox,
                                            image_width=image_width, use_random=use_random)

        # 仅 RA+LA：LA 下方估计
        if ra_bbox is not None and la_bbox is not None:
            return self.infer_lv_from_ra_la(ra_bbox, la_bbox, use_random=use_random)

        return None



    def infer_septum_mask_from_lv_rv_masks(
        self,
        lv_mask: torch.Tensor,
        rv_mask: torch.Tensor,
        thickness_ratio: float = 0.15,
        use_random: bool = True,
        random_range: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """
        从 LV 和 RV mask 推理室间隔 (Interventricular Septum) mask
        
        解析：室间隔是 LV 和 RV 之间的心肌壁
        - 找到 LV 和 RV 各自的边界框
        - 计算两个边界框之间最近的边对
        - 在最近的边之间生成室间隔 mask
        
        Args:
            lv_mask: 左心室 mask (H, W)
            rv_mask: 右心室 mask (H, W)
            thickness_ratio: 室间隔厚度相对于 LV 宽度的比例
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±10% 的厚度变化）
        
        Returns:
            推理出的室间隔 mask (H, W) bool
        """
        if lv_mask is None or rv_mask is None:
            return None
        
        if lv_mask.dim() == 3:
            lv_mask = lv_mask.squeeze(0)
        if rv_mask.dim() == 3:
            rv_mask = rv_mask.squeeze(0)
        
        lv_bool = lv_mask.bool() if lv_mask.dtype != torch.bool else lv_mask
        rv_bool = rv_mask.bool() if rv_mask.dtype != torch.bool else rv_mask
        H, W = lv_bool.shape
        device = lv_bool.device
        
        lv_ys = torch.where(lv_bool.any(dim=1))[0]
        lv_xs = torch.where(lv_bool.any(dim=0))[0]
        rv_ys = torch.where(rv_bool.any(dim=1))[0]
        rv_xs = torch.where(rv_bool.any(dim=0))[0]
        
        if len(lv_ys) == 0 or len(lv_xs) == 0 or len(rv_ys) == 0 or len(rv_xs) == 0:
            return None
        
        lv_x1, lv_x2 = lv_xs.min().item(), lv_xs.max().item()
        lv_y1, lv_y2 = lv_ys.min().item(), lv_ys.max().item()
        rv_x1, rv_x2 = rv_xs.min().item(), rv_xs.max().item()
        rv_y1, rv_y2 = rv_ys.min().item(), rv_ys.max().item()
        
        actual_thickness_ratio = thickness_ratio
        if use_random:
            actual_thickness_ratio = thickness_ratio * (1.0 + (torch.rand(1, device=device).item() - 0.5) * 2 * random_range)
            actual_thickness_ratio = max(0.05, min(0.3, actual_thickness_ratio))
        
        return self._infer_septum_from_bboxes_impl(
            H, W, device,
            lv_x1, lv_y1, lv_x2, lv_y2,
            rv_x1, rv_y1, rv_x2, rv_y2,
            actual_thickness_ratio,
            min_aspect_ratio=0.25,
        )

    def _infer_septum_from_bboxes_impl(
        self,
        H: int, W: int, device,
        lv_x1: int, lv_y1: int, lv_x2: int, lv_y2: int,
        rv_x1: int, rv_y1: int, rv_x2: int, rv_y2: int,
        thickness_ratio: float,
        min_aspect_ratio: float = 0.25,
    ) -> Optional[torch.Tensor]:
        """
        根据两个 bbox 的最近边推理室间隔 mask
        
        检查四对边：
        1. LV右边 vs RV左边 (水平方向)
        2. LV左边 vs RV右边 (水平方向)
        3. LV下边 vs RV上边 (垂直方向)
        4. LV上边 vs RV下边 (垂直方向)
        
        Args:
            min_aspect_ratio: 短边相对于长边的最小比例，默认 0.25 (即 1:4)
                              如果短边/长边 < min_aspect_ratio，则扩展短边
        """
        lv_width = lv_x2 - lv_x1
        lv_height = lv_y2 - lv_y1
        thickness = max(3, int(min(lv_width, lv_height) * thickness_ratio))
        
        distances = []
        
        if rv_x1 >= lv_x2:
            distances.append(("h_right_left", rv_x1 - lv_x2, lv_x2, rv_x1))
        
        if lv_x1 >= rv_x2:
            distances.append(("h_left_right", lv_x1 - rv_x2, rv_x2, lv_x1))
        
        if rv_y1 >= lv_y2:
            distances.append(("v_bottom_top", rv_y1 - lv_y2, lv_y2, rv_y1))
        
        if lv_y1 >= rv_y2:
            distances.append(("v_top_bottom", lv_y1 - rv_y2, rv_y2, lv_y1))
        
        if not distances:
            distances.append(("h_right_left", abs(rv_x1 - lv_x2), min(lv_x2, rv_x1), max(lv_x2, rv_x1)))
            distances.append(("h_left_right", abs(lv_x1 - rv_x2), min(lv_x1, rv_x2), max(lv_x1, rv_x2)))
            distances.append(("v_bottom_top", abs(rv_y1 - lv_y2), min(lv_y2, rv_y1), max(lv_y2, rv_y1)))
            distances.append(("v_top_bottom", abs(lv_y1 - rv_y2), min(lv_y1, rv_y2), max(lv_y1, rv_y2)))
        
        distances.sort(key=lambda x: x[1])
        
        best_direction, best_dist, edge1, edge2 = distances[0]
        
        septum_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        
        if best_direction.startswith("h"):
            y1 = max(lv_y1, rv_y1)
            y2 = min(lv_y2, rv_y2)
            if y2 <= y1:
                y1 = min(lv_y1, rv_y1)
                y2 = max(lv_y2, rv_y2)
            
            x1 = max(0, int(edge1))
            x2 = min(W, int(edge2))
            y1 = max(0, int(y1))
            y2 = min(H, int(y2))
            
            if x2 <= x1:
                mid = (x1 + x2) // 2
                x1 = max(0, mid - thickness // 2)
                x2 = min(W, mid + thickness // 2)
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 0 and height > 0:
                short_side = min(width, height)
                long_side = max(width, height)
                
                if short_side / long_side < min_aspect_ratio:
                    min_short = int(long_side * min_aspect_ratio)
                    if width < height:
                        mid_x = (x1 + x2) // 2
                        x1 = max(0, mid_x - min_short // 2)
                        x2 = min(W, mid_x + min_short // 2)
                    else:
                        mid_y = (y1 + y2) // 2
                        y1 = max(0, mid_y - min_short // 2)
                        y2 = min(H, mid_y + min_short // 2)
            
            if x2 > x1 and y2 > y1:
                septum_mask[y1:y2, x1:x2] = True
        else:
            x1 = max(lv_x1, rv_x1)
            x2 = min(lv_x2, rv_x2)
            if x2 <= x1:
                x1 = min(lv_x1, rv_x1)
                x2 = max(lv_x2, rv_x2)
            
            y1 = max(0, int(edge1))
            y2 = min(H, int(edge2))
            x1 = max(0, int(x1))
            x2 = min(W, int(x2))
            
            if y2 <= y1:
                mid = (y1 + y2) // 2
                y1 = max(0, mid - thickness // 2)
                y2 = min(H, mid + thickness // 2)
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 0 and height > 0:
                short_side = min(width, height)
                long_side = max(width, height)
                
                if short_side / long_side < min_aspect_ratio:
                    min_short = int(long_side * min_aspect_ratio)
                    if width < height:
                        mid_x = (x1 + x2) // 2
                        x1 = max(0, mid_x - min_short // 2)
                        x2 = min(W, mid_x + min_short // 2)
                    else:
                        mid_y = (y1 + y2) // 2
                        y1 = max(0, mid_y - min_short // 2)
                        y2 = min(H, mid_y + min_short // 2)
            
            if y2 > y1 and x2 > x1:
                septum_mask[y1:y2, x1:x2] = True
        
        return septum_mask if septum_mask.any() else None

    def generate_ellipse_mask_from_bbox(
        self,
        bbox: torch.Tensor,
        H: int,
        W: int,
        use_random: bool = True,
        random_range: float = 0.05,
    ) -> Optional[torch.Tensor]:
        """
        从 bbox 生成椭圆 mask
        
        Args:
            bbox: bbox [x1, y1, x2, y2] 归一化坐标
            H, W: 图像尺寸
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±5% 的尺寸和位置变化）
        
        Returns:
            椭圆 mask (H, W) bool
        """
        if bbox is None:
            return None
        
        device = bbox.device
        
        x1 = bbox[0].item() * W
        y1 = bbox[1].item() * H
        x2 = bbox[2].item() * W
        y2 = bbox[3].item() * H
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        rx = (x2 - x1) / 2
        ry = (y2 - y1) / 2
        
        if rx <= 0 or ry <= 0:
            return None
        
        if use_random:
            scale_factor = 1.0 + (torch.rand(1, device=device).item() - 0.5) * 2 * random_range
            scale_factor = max(0.9, min(1.1, scale_factor))
            rx = rx * scale_factor
            ry = ry * scale_factor
            
            offset_x = (torch.rand(1, device=device).item() - 0.5) * 2 * random_range * rx
            offset_y = (torch.rand(1, device=device).item() - 0.5) * 2 * random_range * ry
            cx = cx + offset_x
            cy = cy + offset_y
        
        y_coords = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
        
        ellipse = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2 <= 1
        
        return ellipse

    def infer_interatrial_septum_mask_from_la_ra_masks(
        self,
        la_mask: torch.Tensor,
        ra_mask: torch.Tensor,
        thickness_ratio: float = 0.15,
        use_random: bool = True,
        random_range: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """
        从 LA 和 RA mask 推理房间隔 (Interatrial Septum) mask
        
        房间隔位于 LA 和 RA 之间
        
        Args:
            la_mask: 左心房 mask (H, W)
            ra_mask: 右心房 mask (H, W)
            thickness_ratio: 房间隔厚度相对于 LA 尺寸的比例
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±10% 的厚度变化）
        
        Returns:
            推理出的房间隔 mask (H, W) bool
        """
        if la_mask is None or ra_mask is None:
            return None
        
        if la_mask.dim() == 3:
            la_mask = la_mask.squeeze(0)
        if ra_mask.dim() == 3:
            ra_mask = ra_mask.squeeze(0)
        
        la_bool = la_mask.bool() if la_mask.dtype != torch.bool else la_mask
        ra_bool = ra_mask.bool() if ra_mask.dtype != torch.bool else ra_mask
        H, W = la_bool.shape
        device = la_bool.device
        
        la_ys = torch.where(la_bool.any(dim=1))[0]
        la_xs = torch.where(la_bool.any(dim=0))[0]
        ra_ys = torch.where(ra_bool.any(dim=1))[0]
        ra_xs = torch.where(ra_bool.any(dim=0))[0]
        
        if len(la_ys) == 0 or len(la_xs) == 0 or len(ra_ys) == 0 or len(ra_xs) == 0:
            return None
        
        la_x1, la_x2 = la_xs.min().item(), la_xs.max().item()
        la_y1, la_y2 = la_ys.min().item(), la_ys.max().item()
        ra_x1, ra_x2 = ra_xs.min().item(), ra_xs.max().item()
        ra_y1, ra_y2 = ra_ys.min().item(), ra_ys.max().item()
        
        actual_thickness_ratio = thickness_ratio
        if use_random:
            actual_thickness_ratio = thickness_ratio * (1.0 + (torch.rand(1, device=device).item() - 0.5) * 2 * random_range)
            actual_thickness_ratio = max(0.05, min(0.3, actual_thickness_ratio))
        
        return self._infer_septum_from_bboxes_impl(
            H, W, device,
            la_x1, la_y1, la_x2, la_y2,
            ra_x1, ra_y1, ra_x2, ra_y2,
            actual_thickness_ratio,
            min_aspect_ratio=0.25,
        )

    def infer_mitral_valve_mask_from_lv_la_masks(
        self,
        lv_mask: torch.Tensor,
        la_mask: torch.Tensor,
        thickness_ratio: float = 0.1,
        use_random: bool = True,
        random_range: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """
        从 LV 和 LA mask 推理二尖瓣 (Mitral Valve) mask
        
        二尖瓣位于 LV 和 LA 之间
        
        Args:
            lv_mask: 左心室 mask (H, W)
            la_mask: 左心房 mask (H, W)
            thickness_ratio: 二尖瓣厚度相对于 LV 尺寸的比例
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±10% 的厚度变化）
        
        Returns:
            推理出的二尖瓣 mask (H, W) bool
        """
        if lv_mask is None or la_mask is None:
            return None
        
        if lv_mask.dim() == 3:
            lv_mask = lv_mask.squeeze(0)
        if la_mask.dim() == 3:
            la_mask = la_mask.squeeze(0)
        
        lv_bool = lv_mask.bool() if lv_mask.dtype != torch.bool else lv_mask
        la_bool = la_mask.bool() if la_mask.dtype != torch.bool else la_mask
        H, W = lv_bool.shape
        device = lv_bool.device
        
        lv_ys = torch.where(lv_bool.any(dim=1))[0]
        lv_xs = torch.where(lv_bool.any(dim=0))[0]
        la_ys = torch.where(la_bool.any(dim=1))[0]
        la_xs = torch.where(la_bool.any(dim=0))[0]
        
        if len(lv_ys) == 0 or len(lv_xs) == 0 or len(la_ys) == 0 or len(la_xs) == 0:
            return None
        
        lv_x1, lv_x2 = lv_xs.min().item(), lv_xs.max().item()
        lv_y1, lv_y2 = lv_ys.min().item(), lv_ys.max().item()
        la_x1, la_x2 = la_xs.min().item(), la_xs.max().item()
        la_y1, la_y2 = la_ys.min().item(), la_ys.max().item()
        
        actual_thickness_ratio = thickness_ratio
        if use_random:
            actual_thickness_ratio = thickness_ratio * (1.0 + (torch.rand(1, device=device).item() - 0.5) * 2 * random_range)
            actual_thickness_ratio = max(0.03, min(0.2, actual_thickness_ratio))
        
        return self._infer_septum_from_bboxes_impl(
            H, W, device,
            lv_x1, lv_y1, lv_x2, lv_y2,
            la_x1, la_y1, la_x2, la_y2,
            actual_thickness_ratio,
            min_aspect_ratio=0.25,
        )

    def infer_tricuspid_valve_mask_from_rv_ra_masks(
        self,
        rv_mask: torch.Tensor,
        ra_mask: torch.Tensor,
        thickness_ratio: float = 0.1,
        use_random: bool = True,
        random_range: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """
        从 RV 和 RA mask 推理三尖瓣 (Tricuspid Valve) mask
        
        三尖瓣位于 RV 和 RA 之间
        
        Args:
            rv_mask: 右心室 mask (H, W)
            ra_mask: 右心房 mask (H, W)
            thickness_ratio: 三尖瓣厚度相对于 RV 尺寸的比例
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±10% 的厚度变化）
        
        Returns:
            推理出的三尖瓣 mask (H, W) bool
        """
        if rv_mask is None or ra_mask is None:
            return None
        
        if rv_mask.dim() == 3:
            rv_mask = rv_mask.squeeze(0)
        if ra_mask.dim() == 3:
            ra_mask = ra_mask.squeeze(0)
        
        rv_bool = rv_mask.bool() if rv_mask.dtype != torch.bool else rv_mask
        ra_bool = ra_mask.bool() if ra_mask.dtype != torch.bool else ra_mask
        H, W = rv_bool.shape
        device = rv_bool.device
        
        rv_ys = torch.where(rv_bool.any(dim=1))[0]
        rv_xs = torch.where(rv_bool.any(dim=0))[0]
        ra_ys = torch.where(ra_bool.any(dim=1))[0]
        ra_xs = torch.where(ra_bool.any(dim=0))[0]
        
        if len(rv_ys) == 0 or len(rv_xs) == 0 or len(ra_ys) == 0 or len(ra_xs) == 0:
            return None
        
        rv_x1, rv_x2 = rv_xs.min().item(), rv_xs.max().item()
        rv_y1, rv_y2 = rv_ys.min().item(), rv_ys.max().item()
        ra_x1, ra_x2 = ra_xs.min().item(), ra_xs.max().item()
        ra_y1, ra_y2 = ra_ys.min().item(), ra_ys.max().item()
        
        actual_thickness_ratio = thickness_ratio
        if use_random:
            actual_thickness_ratio = thickness_ratio * (1.0 + (torch.rand(1, device=device).item() - 0.5) * 2 * random_range)
            actual_thickness_ratio = max(0.03, min(0.2, actual_thickness_ratio))
        
        return self._infer_septum_from_bboxes_impl(
            H, W, device,
            rv_x1, rv_y1, rv_x2, rv_y2,
            ra_x1, ra_y1, ra_x2, ra_y2,
            actual_thickness_ratio,
            min_aspect_ratio=0.25,
        )



    def infer_myo_mask_from_lv_mask(
        self,
        lv_mask: torch.Tensor,
        thickness_ratio: float = 0.25,
        bottom_exclude_ratio: float = 0.0,
        use_random: bool = True,
        random_range: float = 0.1,
    ) -> Optional[torch.Tensor]:
        """
        从左心室 (LV) mask 推理心肌 (Myocardium) mask
        
        解析：Myo 是 LV 的外扩区域（心肌壁）
        - 对 LV mask 进行膨胀操作得到外扩区域
        - Myo = 膨胀后的区域 - 原始 LV
        - 四周都保留 Myo（不排除底部，因为图像旋转时可能导致错误截断）
        
        Args:
            lv_mask: 左心室 mask (H, W) 或 (1, H, W)，bool 或 float
            thickness_ratio: 心肌厚度相对于 LV 尺寸的比例
            bottom_exclude_ratio: 底部排除区域占 LV 高度的比例（已弃用，保留参数兼容性）
            use_random: 是否添加随机扰动
            random_range: 随机扰动范围（±10% 的厚度变化）
        
        Returns:
            推理出的 Myo mask (H, W) bool，若无法推理则返回 None
        """
        if lv_mask is None:
            return None
        
        if lv_mask.dim() == 3:
            lv_mask = lv_mask.squeeze(0)
        
        lv_bool = lv_mask.bool() if lv_mask.dtype != torch.bool else lv_mask
        H, W = lv_bool.shape
        device = lv_bool.device
        
        if not lv_bool.any():
            return None
        
        lv_ys = torch.where(lv_bool.any(dim=1))[0]
        lv_xs = torch.where(lv_bool.any(dim=0))[0]
        if len(lv_ys) == 0 or len(lv_xs) == 0:
            return None
        
        lv_h = (lv_ys.max() - lv_ys.min() + 1).float()
        lv_w = (lv_xs.max() - lv_xs.min() + 1).float()
        avg_size = (lv_h + lv_w) / 2
        
        actual_thickness_ratio = thickness_ratio
        if use_random:
            actual_thickness_ratio = thickness_ratio * (1.0 + (torch.rand(1, device=device).item() - 0.5) * 2 * random_range)
            actual_thickness_ratio = max(0.05, min(0.3, actual_thickness_ratio))
        
        thickness = max(2, int(avg_size * actual_thickness_ratio))
        
        lv_dilated = self._dilate_mask(lv_bool, iterations=thickness)
        
        myo_mask = lv_dilated & ~lv_bool
        
        return myo_mask if myo_mask.any() else None

    def _dilate_mask(self, mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
        """
        对 mask 进行膨胀操作
        
        Args:
            mask: 输入 mask (H, W) bool
            iterations: 膨胀迭代次数
        
        Returns:
            膨胀后的 mask (H, W) bool
        """
        result = mask.clone()
        H, W = mask.shape
        device = mask.device
        
        for _ in range(iterations):
            padded = torch.zeros(H + 2, W + 2, dtype=torch.bool, device=device)
            padded[1:-1, 1:-1] = result
            
            dilated = (
                padded[1:-1, 1:-1] |
                padded[:-2, 1:-1] |
                padded[2:, 1:-1] |
                padded[1:-1, :-2] |
                padded[1:-1, 2:]
            )
            result = dilated
        
        return result


