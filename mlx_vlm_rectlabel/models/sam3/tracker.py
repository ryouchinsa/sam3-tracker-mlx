"""SAM3 Tracker: SAM2-style memory-based tracker for video segmentation.

Weight keys: tracker_model.*, tracker_neck.*
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import TrackerConfig
from .sam_components import SAMMaskDecoder, SAMPromptEncoder


class TrackerModel(nn.Module):
    """
    Weight keys: tracker_model.*
    """
    def __init__(self, config: TrackerConfig):
        super().__init__()
        self.backbone_feature_sizes = config.vision_config.backbone_feature_sizes
        self.prompt_encoder = SAMPromptEncoder(config.prompt_encoder_config)
        self.mask_decoder = SAMMaskDecoder(config.mask_decoder_config)

    def decode_mask(
        self,
        image_embeddings: mx.array,
        high_res_features: List[mx.array],
        prompt_points: Optional[Tuple[mx.array, mx.array]] = None,
        prompt_boxes: Optional[mx.array] = None,
        prompt_masks: Optional[mx.array] = None,
        multimask_output: bool = True,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        sparse_emb, dense_emb = self.prompt_encoder(
            input_points=prompt_points,
            input_boxes=prompt_boxes,
            input_masks=prompt_masks,
        )
        batch_size = image_embeddings.shape[0]
        image_pe = self.prompt_encoder.get_dense_pe()
        image_pe = mx.repeat(image_pe, batch_size, axis=0)
        masks, iou_pred, obj_score = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask_output,
            high_res_features=high_res_features,
        )
        return masks, iou_pred, obj_score

