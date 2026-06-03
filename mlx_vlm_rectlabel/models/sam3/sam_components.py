"""SAM2-style components for the tracker: PromptEncoder, MaskDecoder, TwoWayTransformer.

Weight keys: tracker_model.prompt_encoder.*, tracker_model.mask_decoder.*
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import PromptEncoderConfig, TrackerMaskDecoderConfig


class OutputMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, sigmoid_output: bool = False,):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.layers = [nn.Linear(hidden_dim, hidden_dim)]
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.sigmoid_output = sigmoid_output

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.proj_in(x))
        for layer in self.layers:
            x = nn.relu(layer(x))
        x = self.proj_out(x)
        if self.sigmoid_output:
            x = mx.sigmoid(x)
        return x

class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, act: str = "relu"):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, input_dim)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_in(x)
        if self.act == "gelu":
            x = nn.gelu(x)
        else:
            x = nn.relu(x)
        return self.proj_out(x)


class LayerNorm2d(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) - channel last in MLX
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SAMAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        downsample_rate: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        internal_dim = hidden_size // downsample_rate
        self.head_dim = internal_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, internal_dim)
        self.k_proj = nn.Linear(hidden_size, internal_dim)
        self.v_proj = nn.Linear(hidden_size, internal_dim)
        self.o_proj = nn.Linear(internal_dim, hidden_size)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
    ) -> mx.array:
        B, N_q, _, _ = q.shape
        new_shape = (B * N_q, -1, self.num_heads, self.head_dim)

        q = (
            self.q_proj(q)
            .reshape(*new_shape)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(k)
            .reshape(*new_shape)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(v)
            .reshape(*new_shape)
            .transpose(0, 2, 1, 3)
        )

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N_q, -1, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out


class TwoWayAttentionBlock(nn.Module):
    """
    Weight keys per layer:
        self_attn.{q,k,v,o}_proj.{weight,bias}
        cross_attn_token_to_image.{q,k,v,o}_proj.{weight,bias}
        cross_attn_image_to_token.{q,k,v,o}_proj.{weight,bias}
        layer_norm1.{weight,bias}
        layer_norm2.{weight,bias}
        layer_norm3.{weight,bias}
        layer_norm4.{weight,bias}
        mlp.proj_in.{weight,bias}
        mlp.proj_out.{weight,bias}
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False
    ):
        super().__init__()
        self.self_attn = SAMAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.cross_attn_token_to_image = SAMAttention(
            hidden_size, num_heads, downsample_rate=attention_downsample_rate
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLPBlock(hidden_size, mlp_dim, act="relu")
        self.layer_norm3 = nn.LayerNorm(hidden_size)
        self.cross_attn_image_to_token = SAMAttention(
            hidden_size, num_heads, downsample_rate=attention_downsample_rate
        )
        self.layer_norm4 = nn.LayerNorm(hidden_size)
        self.skip_first_layer_pe = skip_first_layer_pe

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        query_pe: mx.array,
        key_pe: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.layer_norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        q = keys + key_pe
        k = queries + query_pe
        attn_out = self.cross_attn_image_to_token(q, k, queries)
        keys = keys + attn_out
        keys = self.layer_norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    """
    Weight keys: tracker_model.mask_decoder.transformer.*
    """
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_dim: int = 2048,
        attention_downsample_rate: int = 2,
    ):
        super().__init__()
        self.layers = [
            TwoWayAttentionBlock(
                hidden_size, num_heads, mlp_dim, attention_downsample_rate, skip_first_layer_pe=(i == 0)
            )
            for i in range(num_layers)
        ]
        self.final_attn_token_to_image = SAMAttention(
            hidden_size, num_heads, downsample_rate=attention_downsample_rate
        )
        self.layer_norm_final_attn = nn.LayerNorm(hidden_size)

    def __call__(
        self,
        image_embedding: mx.array,
        image_pe: mx.array,
        point_embedding: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        B, H, W, D = image_embedding.shape
        image_embedding = image_embedding.reshape(B, 1, H * W, D)
        image_pe = image_pe.reshape(B, 1, H * W, D)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries,
                keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys


class SAMPromptEncoder(nn.Module):
    """
    Weight keys: tracker_model.prompt_encoder.*
    """
    def __init__(self, config: PromptEncoderConfig):
        super().__init__()
        d = config.hidden_size
        image_size = config.image_size
        patch_size = config.patch_size
        self.embed_dim = d
        self.image_embedding_size = (image_size // patch_size, image_size // patch_size)
        self.image_size = image_size
        self.point_embed = nn.Embedding(config.num_point_embeddings, d)
        self.not_a_point_embed = nn.Embedding(1, d)
        self.mask_embed = MaskEmbedConvs(d, config.mask_input_channels)
        self.no_mask_embed = nn.Embedding(1, d)
        self.shared_embedding = PositionalEmbedding(d // 2)

    def get_dense_pe(self) -> mx.array:
        H, W = self.image_embedding_size
        image_pe = self.shared_embedding((H, W))
        image_pe = image_pe.reshape(1, H, W, self.embed_dim)
        return image_pe

    def __call__(
        self,
        input_points: Optional[Tuple[mx.array, mx.array]] = None,
        input_boxes: Optional[mx.array] = None,
        input_masks: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        batch_size = 1
        sparse_embeddings = None
        if input_points is not None:
            coords, labels = input_points
            batch_size = coords.shape[0]
            point_emb = self._embed_points(coords, labels, pad=(input_boxes is None))
            sparse_embeddings = point_emb
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_emb = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_emb
            else:
                sparse_embeddings = mx.concatenate([sparse_embeddings, box_emb], axis=2)
        if input_masks is not None:
            masks = input_masks.transpose(0, 2, 3, 1)
            dense_embeddings = self.mask_embed(masks)
        else:
            H, W = self.image_embedding_size
            dense_embeddings = self.no_mask_embed.weight.reshape(1, 1, 1, -1)
            dense_embeddings = mx.broadcast_to(
                dense_embeddings, (batch_size, H, W, self.embed_dim)
            )
        return sparse_embeddings, dense_embeddings

    def _embed_points(self, coords: mx.array, labels: mx.array, pad: bool) -> mx.array:
        coords = coords + 0.5
        if pad:
            coords = mx.pad(coords, ((0, 0), (0, 0), (0, 1), (0, 0)), constant_values=0)
            labels = mx.pad(labels, ((0, 0), (0, 0), (0, 1)), constant_values=-1)
        
        coords = coords / mx.array([self.image_size, self.image_size], dtype=mx.float32)
        point_emb = self.shared_embedding.forward_with_coords(coords)

        padding_mask = labels == -1
        if padding_mask.any():
            not_a_point = self.not_a_point_embed.weight
            point_emb = mx.where(padding_mask[..., None], not_a_point, point_emb)

        padding_mask = labels != -10
        if padding_mask.any():
            point_emb = mx.where(padding_mask[..., None], point_emb, mx.zeros_like(point_emb))

        point_emb = point_emb + self.point_embed(mx.clip(labels, a_min=0, a_max=None)) * mx.expand_dims((labels >= 0), axis=-1)
        return point_emb

    def _embed_boxes(self, boxes: mx.array) -> mx.array:
        boxes = boxes + 0.5
        coords = boxes.reshape(*boxes.shape[:2], 2, 2)
        coords = mx.pad(coords, ((0, 0), (0, 0), (0, 1), (0, 0)), constant_values=0)
        coords = coords / mx.array([self.image_size, self.image_size], dtype=mx.float32)
        corner_emb = self.shared_embedding.forward_with_coords(coords)
        corner_emb[:, :, 0, :] += self.point_embed.weight[2]
        corner_emb[:, :, 1, :] += self.point_embed.weight[3]
        corner_emb[:, :, 2, :] = mx.broadcast_to(self.not_a_point_embed.weight, corner_emb[:, :, 2, :].shape)
        return corner_emb

class MaskEmbedConvs(nn.Module):
    """
    Weight keys: tracker_model.prompt_encoder.mask_embed.*
    """
    def __init__(self, embed_dim: int, mask_in_chans: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2
        )
        self.conv3 = nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
        self.layer_norm1 = LayerNorm2d(mask_in_chans // 4)
        self.layer_norm2 = LayerNorm2d(mask_in_chans)

    def __call__(self, masks: mx.array) -> mx.array:
        x = self.conv1(masks)
        x = self.layer_norm1(x)
        x = nn.gelu(x)
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = nn.gelu(x)
        x = self.conv3(x)
        return x


class PositionalEmbedding(nn.Module):
    """
    Weight keys: tracker_model.prompt_encoder.shared_embedding.positional_embedding
    """
    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.positional_embedding = mx.zeros((2, num_pos_feats))

    def __call__(self, size: Tuple[int, int]) -> mx.array:
        H, W = size
        grid_y = mx.arange(H).astype(mx.float32) + 0.5
        grid_x = mx.arange(W).astype(mx.float32) + 0.5
        grid_y = grid_y / H
        grid_x = grid_x / W
        gy, gx = mx.meshgrid(grid_y, grid_x, indexing="ij")
        coords = mx.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
        return self.forward_with_coords(coords[None])

    def forward_with_coords(self, coords: mx.array) -> mx.array:
        coords = 2 * coords - 1
        coords = coords @ self.positional_embedding
        coords = 2 * math.pi * coords
        return mx.concatenate([mx.sin(coords), mx.cos(coords)], axis=-1)


class SAMMaskDecoder(nn.Module):
    """
    Weight keys: tracker_model.mask_decoder.*
    """
    def __init__(self, config: TrackerMaskDecoderConfig):
        super().__init__()
        d = config.hidden_size
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.transformer = TwoWayTransformer(
            hidden_size=d,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
            mlp_dim=config.mlp_dim,
            attention_downsample_rate=config.attention_downsample_rate,
        )

        self.iou_token = nn.Embedding(1, d)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, d)
        self.obj_score_token = nn.Embedding(1, d)

        self.output_hypernetworks_mlps = [
            OutputMLP(d, d, d // 8) for _ in range(self.num_mask_tokens)
        ]
        self.iou_prediction_head = OutputMLP(d, d, self.num_mask_tokens, sigmoid_output=True)
        self.pred_obj_score_head = OutputMLP(d, d, 1)

        self.upscale_conv1 = nn.ConvTranspose2d(d, d // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(d // 4, d // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(d // 4)

        self.conv_s0 = nn.Conv2d(d, d // 8, kernel_size=1, bias=True)
        self.conv_s1 = nn.Conv2d(d, d // 4, kernel_size=1, bias=True)

        self.dynamic_multimask_via_stability = config.dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = (
            config.dynamic_multimask_stability_delta
        )
        self.dynamic_multimask_stability_thresh = (
            config.dynamic_multimask_stability_thresh
        )

    def __call__(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
        multimask_output: bool = True,
        high_res_features: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        batch_size, w, h, d = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]

        tokens = mx.concatenate(
            [
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ],
            axis=0,
        )
        tokens = mx.tile(tokens, (batch_size, point_batch_size, 1, 1))
        tokens = mx.concatenate([tokens, sparse_prompt_embeddings], axis=2)

        src = image_embeddings + dense_prompt_embeddings
        src = mx.repeat(src, point_batch_size, axis=0)
        image_pe = mx.repeat(image_pe, point_batch_size, axis=0)

        hs, src = self.transformer(src, image_pe, tokens)
        iou_token_out = hs[:, :, 1, :]
        mask_tokens_out = hs[:, :, 2 : (2 + self.num_mask_tokens), :]
        obj_score_token_out = hs[:, :, 0, :]

        src = src.reshape(batch_size * point_batch_size, w, h, d)

        feat_s0, feat_s1 = high_res_features
        feat_s0 = mx.repeat(feat_s0, point_batch_size, axis=0)
        feat_s1 = mx.repeat(feat_s1, point_batch_size, axis=0)

        upscaled = self.upscale_conv1(src) + feat_s1
        upscaled = self.upscale_layer_norm(upscaled)
        upscaled = nn.gelu(upscaled)
        upscaled = self.upscale_conv2(upscaled)+ feat_s0
        upscaled = nn.gelu(upscaled)

        _, H_up, W_up, C_up = upscaled.shape
        upscaled_flat = upscaled.reshape(batch_size, point_batch_size, H_up * W_up, C_up)
        masks = []
        for i in range(self.num_mask_tokens):
            hyper_out = self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i])
            mask = (upscaled_flat * hyper_out[:, :, None, :]).sum(axis=-1)
            masks.append(mask.reshape(batch_size, point_batch_size, -1, H_up, W_up))
        masks = mx.concatenate(masks, axis=2)

        # upscaled = upscaled.transpose(0, 3, 1, 2)
        # _, C_up, H_up, W_up = upscaled.shape
        # upscaled = upscaled.reshape(B, point_batch_size, C_up, H_up * W_up)
        # hyper_in_list = []
        # for i in range(self.num_mask_tokens):
        #     current_mlp = self.output_hypernetworks_mlps[i]
        #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        # hyper_in = mx.stack(hyper_in_list, axis=2)
        # masks = (hyper_in @ upscaled).reshape(B, point_batch_size, -1, H_up, W_up)

        iou_pred = self.iou_prediction_head(iou_token_out)
        obj_score = self.pred_obj_score_head(obj_score_token_out)
        
        if multimask_output:
            masks = masks[:, :, 1:]  # skip first (low-res) mask
            iou_pred = iou_pred[:, :, 1:]
        else:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)

        return masks, iou_pred, obj_score

    def _get_stability_scores(self, mask_logits):
        mask_logits = mx.flatten(mask_logits, start_axis=-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = mx.sum(mask_logits > stability_delta, axis=-1).astype(mx.float32)
        area_u = mx.sum(mask_logits > -stability_delta, axis=-1).astype(mx.float32)
        stability_scores = mx.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, :, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, :, 1:]
        best_scores_inds = mx.argmax(multimask_iou_scores, axis=-1)  # [B, P]
        best_scores_inds_expanded = best_scores_inds[..., None, None, None] 
        best_scores_inds_expanded = mx.broadcast_to(
            best_scores_inds_expanded, 
            (
                best_scores_inds_expanded.shape[0], 
                best_scores_inds_expanded.shape[1], 
                1, 
                multimask_logits.shape[-2], 
                multimask_logits.shape[-1]
            )
        )
        best_multimask_logits = mx.take_along_axis(multimask_logits, best_scores_inds_expanded, axis=2)  # [B, P, 1, H, W]
        best_multimask_iou_scores = mx.take_along_axis(multimask_iou_scores, mx.expand_dims(best_scores_inds, axis=-1), axis=2)  # [B, P, 1]

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, :, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, :, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = mx.where(
            mx.broadcast_to(is_stable[..., None, None], singlemask_logits.shape),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = mx.where(
            mx.broadcast_to(is_stable, singlemask_iou_scores.shape),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out










































