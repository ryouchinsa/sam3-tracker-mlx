import glob
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


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


def load_model_mlx(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
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
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    return model


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



