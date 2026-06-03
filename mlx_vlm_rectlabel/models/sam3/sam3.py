"""SAM3 Main Model: Combines detector (DETR-based) + tracker (SAM2-based).

Weight keys:
    detector_model.*  -> self.detector_model.*
    tracker_model.*   -> self.tracker_model.*
    tracker_neck.*    -> self.tracker_neck.*
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .tracker import TrackerModel
from .vision import FPNNeck, VisionEncoder
import time

# ---------------------------------------------------------------------------
# Detector Model
# ---------------------------------------------------------------------------


class DetectorModel(nn.Module):
    """SAM3 detection model: vision + text -> DETR -> masks.

    Weight keys: detector_model.*
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        det_cfg = config.detector_config

        # Vision encoder (backbone + FPN neck)
        self.vision_encoder = VisionEncoder(det_cfg.vision_config)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """SAM3 full model: detector + tracker.

    Weight keys:
        detector_model.*  -> self.detector_model.*
        tracker_model.*   -> self.tracker_model.*
        tracker_neck.*    -> self.tracker_neck.*
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # Detector (image segmentation)
        self.detector_model = DetectorModel(config)

        # Tracker (video tracking)
        self.tracker_model = TrackerModel(config.tracker_config)

        # Tracker FPN neck (separate from detector's neck)
        self.tracker_neck = FPNNeck(config.tracker_config.vision_config)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Convert HuggingFace PyTorch weights to MLX format.

        Main conversions:
        1. Conv2d: PyTorch [out, in, H, W] -> MLX [out, H, W, in]
        2. ConvTranspose2d: PyTorch [in, out, H, W] -> MLX [out, H, W, in]
        """
        sanitized = {}

        # Patterns for ConvTranspose2d weights (need transpose(1,2,3,0))
        conv_transpose_patterns = [
            "scale_layers.",  # FPN upsampling layers
            "upscale_conv",  # Tracker mask decoder upscaling
        ]

        # Patterns for special 4D weights that are NOT ConvTranspose2d
        # (regular Conv2d: transpose(0,2,3,1))
        conv2d_patterns = [
            "projection.weight",  # Patch embedding
            "proj1.weight",  # FPN 1x1 conv
            "proj2.weight",  # FPN 3x3 conv
            ".conv.",  # Generic conv layers
            "conv_layers.",  # Pixel decoder convs
            "instance_projection.",  # 1x1 conv
            "semantic_projection.",  # 1x1 conv
            "feature_projection.",  # Memory encoder 1x1
            "final_conv.",  # Mask downsampler final
            "conv_s0.",  # Tracker skip conv
            "conv_s1.",  # Tracker skip conv
            "depthwise_conv.",  # CXBlock depthwise
            "mask_downsample.",  # Mask downsampling
            "conv1.",  # Mask embed convs
            "conv2.",  # Mask embed convs
            "conv3.",  # Mask embed convs
            "boxes_pool_project.",  # Geometry encoder Conv2d
        ]

        # 4D parameters that are NOT convolution weights (skip transposition)
        skip_transpose_patterns = [
            "memory_temporal_positional_encoding",
        ]

        for key, value in weights.items():
            if value.ndim == 4:
                # Skip non-conv 4D parameters
                if any(p in key for p in skip_transpose_patterns):
                    sanitized[key] = value
                    continue

                # 4D tensor: either Conv2d or ConvTranspose2d
                is_conv_transpose = any(p in key for p in conv_transpose_patterns)

                if is_conv_transpose:
                    # PyTorch ConvTranspose2d: (in_ch, out_ch, kH, kW)
                    # MLX ConvTranspose2d: (out_ch, kH, kW, in_ch)
                    value = value.transpose(1, 2, 3, 0)
                else:
                    # PyTorch Conv2d: (out_ch, in_ch, kH, kW)
                    # MLX Conv2d: (out_ch, kH, kW, in_ch)
                    value = value.transpose(0, 2, 3, 1)

            sanitized[key] = value

        return sanitized




















        
