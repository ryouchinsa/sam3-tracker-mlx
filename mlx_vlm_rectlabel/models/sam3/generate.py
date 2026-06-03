import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
import time

class Sam3VideoPredictor:

    def __init__(
        self, 
        model, 
        preprocess_like_torch: bool = False,
    ):
        self.model = model
        self.image_size = 1008
        self.image_mean = [127.5, 127.5, 127.5]
        self.image_std = [127.5, 127.5, 127.5]
        self.preprocess_like_torch = preprocess_like_torch

    def get_feature_maps(
        self, 
        images: [Image.Image]
    ) -> mx.array:
        inputs = self.preprocess_image(images)
        pixel_values = mx.array(inputs["pixel_values"])
        features = self.model.detector_model.vision_encoder.backbone(pixel_values)
        features = self.model.tracker_neck(features)
        features[0] = self.model.tracker_model.mask_decoder.conv_s0(features[0])
        features[1] = self.model.tracker_model.mask_decoder.conv_s1(features[1])
        feature_maps = []
        for idx, feat_size in enumerate(self.model.tracker_model.backbone_feature_sizes):
            feature_maps.append(features[idx])
        mx.eval(feature_maps)
        return feature_maps

    def preprocess_image(
        self,
        images: [Image.Image],
    ) -> Dict[str, np.ndarray]:
        pixel_values = [self._process_single_image(img) for img in images]
        pixel_values = np.stack(pixel_values)
        return {"pixel_values": pixel_values}

    def _process_single_image(
        self, 
        image: Image.Image
    ) -> np.ndarray:
        image = image.convert("RGB")
        if self.preprocess_like_torch:
            import torch
            from torchvision.transforms.v2 import functional as tvF
            image = np.array(image)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)
            image = tvF.resize(image, (self.image_size, self.image_size), interpolation=tvF.InterpolationMode.BILINEAR, antialias=True)
            image = tvF.normalize(image.to(dtype=torch.float32), self.image_mean, self.image_std)
            pixel_values = np.array(image)
            pixel_values = pixel_values.transpose(1, 2, 0)
            return pixel_values
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        pixel_values = np.array(image).astype(np.float32)
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        return pixel_values

    def add_points_boxes(
        self,
        feature_maps: mx.array,
        original_sizes: [Tuple],
        input_points: Optional[np.ndarray] = None,
        input_labels: Optional[np.ndarray] = None,
        input_boxes: Optional[np.ndarray] = None,
        input_masks: Optional[mx.array] = None,
        multimask_output: bool = True,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        prompt_points = None
        if input_points is not None:
            input_points, input_labels = self._pad_points(input_points, input_labels)
            input_points = self._preprocess_points(input_points, original_sizes)
            input_points = mx.array(input_points)
            input_labels = mx.array(input_labels)
            prompt_points = (input_points, input_labels)
        prompt_boxes = None
        if input_boxes is not None:
            input_boxes = self._preprocess_boxes(input_boxes, original_sizes)
            input_boxes = mx.array(input_boxes)
            prompt_boxes=input_boxes
        masks, iou_pred, obj_score = self.model.tracker_model.decode_mask(
            image_embeddings=feature_maps[2],
            high_res_features=[feature_maps[0], feature_maps[1]],
            prompt_points=prompt_points,
            prompt_boxes=prompt_boxes,
            prompt_masks=input_masks,
            multimask_output=multimask_output,
        )
        return masks, iou_pred, obj_score

    def _pad_points(
        self,
        input_points: np.ndarray,
        input_labels: np.ndarray,
    ):
        object_num_max, point_num_max = self._get_object_point_num_max(input_points)
        batch_num = len(input_points)
        for b in range(batch_num):
            object_num = len(input_points[b])
            add_object = object_num_max - object_num
            for a in range(add_object):
                points, labels = self._get_pad_points_labels(point_num_max)
                input_points[b].append(points)
                input_labels[b].append(labels)
            object_num = len(input_points[b])
            for o in range(object_num):
                point_num = len(input_points[b][o])
                add_point = point_num_max - point_num
                for a in range(add_point):
                    input_points[b][o].append([-10, -10])
                    input_labels[b][o].append(-10)
        return input_points, input_labels

    def _get_object_point_num_max(
        self,
        input_points: np.ndarray
    ):
        batch_num = len(input_points)
        object_num_max = 0
        point_num_max = 0
        for b in range(batch_num):
            object_num = len(input_points[b])
            if object_num > object_num_max:
                object_num_max = object_num
            for o in range(object_num):
                point_num = len(input_points[b][o])
                if point_num > point_num_max:
                    point_num_max = point_num
        return object_num_max, point_num_max

    def _get_pad_points_labels(
        self,
        point_num_max
    ):
        points = []
        labels = []
        for a in range(point_num_max):
            points.append([-10, -10])
            labels.append(-10)
        return points, labels

    def _preprocess_points(
        self,
        input_points: np.ndarray,
        original_sizes: [Tuple]
    ) -> np.ndarray:
        batch_num = len(input_points)
        object_num = len(input_points[0])
        point_num = len(input_points[0][0])
        for b in range(batch_num):
            original_size = original_sizes[b]
            for o in range(object_num):
                for n in range(point_num):
                    if input_points[b][o][n][0] != -10:
                        input_points[b][o][n][0] = input_points[b][o][n][0] * self.image_size / original_size[0]
                        input_points[b][o][n][1] = input_points[b][o][n][1] * self.image_size / original_size[1]
        return input_points

    def _preprocess_boxes(
        self,
        input_boxes: np.ndarray,
        original_sizes: [Tuple]
    ) -> np.ndarray:
        batch_num = len(input_boxes)
        box_num = len(input_boxes[0])
        for b in range(batch_num):
            original_size = original_sizes[b]
            for n in range(box_num):
                input_boxes[b][n][0] = input_boxes[b][n][0] * self.image_size / original_size[0]
                input_boxes[b][n][1] = input_boxes[b][n][1] * self.image_size / original_size[1]
                input_boxes[b][n][2] = input_boxes[b][n][2] * self.image_size / original_size[0]
                input_boxes[b][n][3] = input_boxes[b][n][3] * self.image_size / original_size[1]
        return input_boxes






























