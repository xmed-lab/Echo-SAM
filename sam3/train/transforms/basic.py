# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Transforms and data augmentation for both image + bbox.
"""

import math
import random
from typing import Iterable

import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.data_misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "positive_map"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "input_boxes" in target:
        boxes = target["input_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["input_boxes"] = cropped_boxes.reshape(-1, 4)

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "input_boxes" in target:
        boxes = target["input_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["input_boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    if "text_input" in target:
        text_input = (
            target["text_input"]
            .replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )
        target["text_input"] = text_input

    return flipped_image, target


def rotate(image, target, angle):
    w, h = image.size
    rotated_image = F.rotate(image, angle, expand=True, interpolation=F.InterpolationMode.BILINEAR)
    new_w, new_h = rotated_image.size

    target = target.copy()
    target["size"] = torch.tensor([new_h, new_w])

    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = w / 2, h / 2

    def rotate_boxes(boxes):
        x1, y1, x2, y2 = boxes.unbind(1)
        points = torch.stack([
            x1, y1,
            x2, y1,
            x2, y2,
            x1, y2
        ], dim=1).reshape(-1, 4, 2)
        points = points - torch.tensor([[cx, cy]], dtype=torch.float32)
        rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
        rotated_points = points @ rot_matrix.T
        rotated_points = rotated_points + torch.tensor([[new_w / 2, new_h / 2]], dtype=torch.float32)
        x_min = rotated_points[..., 0].min(dim=1)[0]
        y_min = rotated_points[..., 1].min(dim=1)[0]
        x_max = rotated_points[..., 0].max(dim=1)[0]
        y_max = rotated_points[..., 1].max(dim=1)[0]
        rotated_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        max_bounds = torch.tensor([new_w, new_h, new_w, new_h], dtype=torch.float32)
        return rotated_boxes.clamp(0.0, max_bounds)

    if "boxes" in target:
        target["boxes"] = rotate_boxes(target["boxes"])
    if "input_boxes" in target:
        target["input_boxes"] = rotate_boxes(target["input_boxes"])
    if "masks" in target:
        masks = target["masks"]
        num_masks = masks.shape[0]
        masks_pil = []
        for i in range(num_masks):
            mask_np = masks[i].cpu().numpy().astype('uint8') * 255
            masks_pil.append(PIL.Image.fromarray(mask_np, mode='L'))
        rotated_masks = []
        for m_pil in masks_pil:
            rotated_m_pil = F.rotate(m_pil, angle, expand=True, interpolation=F.InterpolationMode.NEAREST)
            rotated_mask_np = np.array(rotated_m_pil)
            rotated_mask = torch.from_numpy(rotated_mask_np).bool()
            rotated_masks.append(rotated_mask)
        target["masks"] = torch.stack(rotated_masks, dim=0).to(masks.device)

    return rotated_image, target


def resize(image, target, size, max_size=None, square=False):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32
        )
        target["boxes"] = scaled_boxes
    if "input_boxes" in target:
        boxes = target["input_boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32
        )
        target["input_boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0]
            > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    else:
        # left, top, right, bottom
        padded_image = F.pad(image, (padding[0], padding[1], padding[2], padding[3]))
    if target is None:
        return padded_image, None
    target = target.copy()

    w, h = padded_image.size

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])
    if "boxes" in target and len(padding) == 4:
        boxes = target["boxes"]
        boxes = boxes + torch.as_tensor(
            [padding[0], padding[1], padding[0], padding[1]], dtype=torch.float32
        )
        target["boxes"] = boxes

    if "input_boxes" in target and len(padding) == 4:
        boxes = target["input_boxes"]
        boxes = boxes + torch.as_tensor(
            [padding[0], padding[1], padding[0], padding[1]], dtype=torch.float32
        )
        target["input_boxes"] = boxes

    if "masks" in target:
        if len(padding) == 2:
            target["masks"] = torch.nn.functional.pad(
                target["masks"], (0, padding[0], 0, padding[1])
            )
        else:
            target["masks"] = torch.nn.functional.pad(
                target["masks"], (padding[0], padding[2], padding[1], padding[3])
            )
    return padded_image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes  # if True we can't crop a box out

    def __call__(self, img: PIL.Image.Image, target: dict):
        init_boxes = len(target["boxes"])
        init_boxes_tensor = target["boxes"].clone()
        if self.respect_boxes and init_boxes > 0:
            minW, minH, maxW, maxH = (
                min(img.width, self.min_size),
                min(img.width, self.min_size),
                min(img.width, self.max_size),
                min(img.height, self.max_size),
            )
            minX, minY = (
                target["boxes"][:, 0].max().item() + 10.0,
                target["boxes"][:, 1].max().item() + 10.0,
            )
            minX = min(img.width, minX)
            minY = min(img.height, minY)
            maxX, maxY = (
                target["boxes"][:, 2].min().item() - 10,
                target["boxes"][:, 3].min().item() - 10,
            )
            maxX = max(0.0, maxX)
            maxY = max(0.0, maxY)
            minW = max(minW, minX - maxX)
            minH = max(minH, minY - maxY)
            w = random.uniform(minW, max(minW, maxW))
            h = random.uniform(minH, max(minH, maxH))
            if minX > maxX:
                # i = random.uniform(max(0, minX - w + 1), max(maxX, max(0, minX - w + 1)))
                i = random.uniform(max(0, minX - w), max(maxX, max(0, minX - w)))
            else:
                i = random.uniform(
                    max(0, minX - w + 1), max(maxX - 1, max(0, minX - w + 1))
                )
            if minY > maxY:
                # j = random.uniform(max(0, minY - h + 1), max(maxY, max(0, minY - h + 1)))
                j = random.uniform(max(0, minY - h), max(maxY, max(0, minY - h)))
            else:
                j = random.uniform(
                    max(0, minY - h + 1), max(maxY - 1, max(0, minY - h + 1))
                )
            result_img, result_target = crop(img, target, [j, i, h, w])
            assert len(result_target["boxes"]) == init_boxes, (
                f"img_w={img.width}\timg_h={img.height}\tminX={minX}\tminY={minY}\tmaxX={maxX}\tmaxY={maxY}\tminW={minW}\tminH={minH}\tmaxW={maxW}\tmaxH={maxH}\tw={w}\th={h}\ti={i}\tj={j}\tinit_boxes={init_boxes_tensor}\tresults={result_target['boxes']}"
            )

            return result_img, result_target
        else:
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, (h, w))
            result_img, result_target = crop(img, target, region)
            return result_img, result_target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomRotate:
    def __init__(self, degrees, p=0.5):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            return rotate(img, target, angle)
        return img, target


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.8):
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = self.color_jitter(img)
        return img, target


class RandomGrayscale:
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.to_grayscale(img, num_output_channels=3)
        return img, target


class RandomGaussianBlur:
    def __init__(self, kernel_size=(5, 9), sigma=(0.1, 5.0), p=0.5):
        self.kernel_size = tuple(kernel_size) if hasattr(kernel_size, "__iter__") and not isinstance(kernel_size, (int, float)) else kernel_size
        self.sigma = tuple(sigma) if hasattr(sigma, "__iter__") and not isinstance(sigma, (int, float)) else sigma
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            k_size = random.choice(self.kernel_size) if isinstance(self.kernel_size, tuple) else self.kernel_size
            sigma_val = random.uniform(self.sigma[0], self.sigma[1]) if isinstance(self.sigma, tuple) else self.sigma
            img = F.gaussian_blur(img, kernel_size=k_size, sigma=sigma_val)
        return img, target


class RandomSolarize:
    def __init__(self, threshold=128, p=0.2):
        self.threshold = threshold
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.solarize(img, threshold=self.threshold)
        return img, target


class RandomResize:
    def __init__(self, sizes, max_size=None, square=False):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, square=self.square)


class RandomPad:
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class PadToSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        w, h = img.size
        pad_x = self.size - w
        pad_y = self.size - h
        assert pad_x >= 0 and pad_y >= 0
        pad_left = random.randint(0, pad_x)
        pad_right = pad_x - pad_left
        pad_top = random.randint(0, pad_y)
        pad_bottom = pad_y - pad_top
        return pad(img, target, (pad_left, pad_top, pad_right, pad_bottom))


class Identity:
    def __call__(self, img, target):
        return img, target


class RandomSelect:
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1=None, transforms2=None, p=0.5):
        self.transforms1 = transforms1 or Identity()
        self.transforms2 = transforms2 or Identity()
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "input_boxes" in target:
            boxes = target["input_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["input_boxes"] = boxes
        return image, target


class RemoveDifficult:
    def __init__(self, enabled=False):
        self.remove_difficult = enabled

    def __call__(self, image, target=None):
        if target is None:
            return image, None
        target = target.copy()
        keep = ~target["iscrowd"].to(torch.bool) | (not self.remove_difficult)
        if "boxes" in target:
            target["boxes"] = target["boxes"][keep]
        target["labels"] = target["labels"][keep]
        target["iscrowd"] = target["iscrowd"][keep]
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def get_random_resize_scales(size, min_size, rounded):
    stride = 128 if rounded else 32
    min_size = int(stride * math.ceil(min_size / stride))
    scales = list(range(min_size, size + 1, stride))
    return scales


def get_random_resize_max_size(size, ratio=5 / 3):
    max_size = round(ratio * size)
    return max_size
