import glob
import importlib
import inspect
import json
import logging
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import requests
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_map
from PIL import Image, ImageOps


# Constants
MODEL_REMAPPING = {
    "llava_qwen2": "fastvlm",  # Apple's FastVLM, note it's different to the one below
    "llava-qwen2": "llava_bunny",
    "bunny-llama": "llava_bunny",
    "lfm2-vl": "lfm2_vl",
    "cohere2_vision": "aya_vision",
    "jvlm": "jina_vlm",
    "phi4-siglip": "phi4_siglip",
    "sam3_video": "sam3",
    "sam3.1_video": "sam3_1",
    "granite-vision": "granite_vision",
    "granite4-vision": "granite4_vision",
    "granite4_vision": "granite4_vision",
    "rf-detr": "rfdetr",
    "falcon-perception": "falcon_perception",
}

MAX_FILE_SIZE_GB = 5

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def get_model_and_args(config: dict):
    """
    Retrieve the model object based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"].lower()

    model_type = MODEL_REMAPPING.get(model_type, model_type)

    try:
        arch = importlib.import_module(f"mlx_vlm_rectlabel.models.{model_type}")
    except ImportError as e:
        msg = f"Model type {model_type} not supported. Error: {e}"
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


def get_model_path(
    path_or_hf_repo: str, revision: Optional[str] = None, force_download: bool = False
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jinja",
                ],
                force_download=force_download,
            )
        )
    return model_path


def load_model(model_path: Path, lazy: bool = False, convert: bool = False, **kwargs) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.
        quantize_activations (bool, optional): If True, convert QuantizedLinear layers
            to QQLinear layers for activation quantization. Only supported for models
            quantized with 'nvfp4' or 'mxfp8' modes. Default: ``False``.
        quantize_activations (bool, optional): If True, convert QuantizedLinear layers
            to QQLinear layers for activation quantization. Only supported for models
            quantized with 'nvfp4' or 'mxfp8' modes. Default: ``False``.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path, **kwargs)

    # Find all .safetensors files in the model_path, excluding consolidated model weights
    weight_files = [
        wf
        for wf in glob.glob(str(model_path / "*.safetensors"))
        if not wf.endswith("consolidated.safetensors")
    ]

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        message = f"""
No safetensors found in {model_path}
Create safetensors using the following code:
```
from transformers import AutoModelForCausalLM, AutoProcessor

model_id= "<huggingface_model_id>"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("<local_dir>")
processor.save_pretrained("<local_dir>")
```
Then use the <local_dir> as the --hf-path in the convert script.
```
python -m mlx_vlm.convert --hf-path <local_dir> --mlx-path <mlx_dir>
```
        """
        raise FileNotFoundError(message)

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    import safetensors

    with safetensors.safe_open(weight_files[0], framework="np") as f:
        is_mlx_format = f.metadata() and f.metadata().get("format") == "mlx"

    model_class, _ = get_model_and_args(config=config)

    # Initialize text and vision configs if not present
    # config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    # config.setdefault("audio_config", {})

    # Initialize model config and update it with module configs
    model_config = model_class.ModelConfig.from_dict(config)
    # modules = ["text", "vision", "perceiver", "projector", "audio"]
    modules = ["vision"]
    model_config = update_module_configs(model_config, model_class, config, modules)

    model = model_class.Model(model_config)

    if not is_mlx_format:
        # Sanitize weights
        weights = sanitize_weights(model, weights)

        if hasattr(model, "thinker") and hasattr(model.thinker, "sanitize"):
            weights = sanitize_weights(model.thinker, weights)
            weights = sanitize_weights(model.thinker.vision_tower, weights)
            weights = sanitize_weights(model.thinker.audio_tower, weights)
            weights = sanitize_weights(model.thinker.language_model, weights)
            weights = sanitize_weights(model.code2wav, weights)
            weights = sanitize_weights(model.talker, weights)
        else:
            weights = sanitize_weights(
                model_class.VisionModel, weights, model_config.vision_config
            )
            # weights = sanitize_weights(
            #     model_class.LanguageModel, weights, model_config.text_config
            # )
            # if hasattr(model_class, "AudioModel"):
            #     weights = sanitize_weights(
            #         model_class.AudioModel, weights, model_config.audio_config
            #     )

    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    if convert:
        dtype = "float16"
        save_path = "sam3_" + dtype
        print("[INFO] Using dtype:", dtype, save_path)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        from mlx.utils import tree_map_with_path
        model.update(tree_map_with_path(set_dtype, model.parameters()))
        save_weights(save_path, model, donate_weights=True)

    return model


def sanitize_weights(model_obj, weights, config=None):
    """Helper function to sanitize weights if the model has a sanitize method"""
    if hasattr(model_obj, "sanitize"):
        if config is not None:
            model_obj = model_obj(config)
        weights = model_obj.sanitize(weights)
    return weights


def update_module_configs(model_config, model_class, config, modules):
    """Updates configuration for model modules like text and vision modules.

    Args:
        model_config: The model configuration object that will be updated
        model_class: The model class containing component config classes
        config: Dictionary containing configuration parameters
        modules: List of module names to update configs for (e.g. ["text", "vision"])

    Returns:
        The updated model_config object
    """
    for config_name in modules:
        config_attr = f"{config_name}_config"
        if hasattr(model_config, config_attr) and config.get(config_attr) is not None:
            config_class = getattr(model_class, f"{config_name.title()}Config")
            setattr(
                model_config, config_attr, config_class.from_dict(config[config_attr])
            )
    return model_config


def load_config(model_path: Union[str, Path], **kwargs) -> dict:
    """Load model configuration from a path or Hugging Face repo.

    Args:
        model_path: Local path or Hugging Face repo ID to load config from
        **kwargs: Additional keyword arguments to pass to the config loader

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found at the path
    """
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    try:
        with open(model_path / "config.json", encoding="utf-8") as f:
            config = json.load(f)

        generation_config_file = model_path / "generation_config.json"
        if generation_config_file.exists():
            generation_config = {}
            try:
                with open(generation_config_file, "r") as f:
                    generation_config = json.load(f)
            except json.JSONDecodeError:
                pass

            if eos_token_id := generation_config.get("eos_token_id", False):
                config["eos_token_id"] = eos_token_id

        return config

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config not found at {model_path}") from exc


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_weights(
    save_path: Union[str, Path],
    model: nn.Module,
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    weights = dict(tree_flatten(model.parameters()))

    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        model.update(tree_map(lambda _: mx.array([]), model.parameters()))

    weights.clear()
    del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        del shard
