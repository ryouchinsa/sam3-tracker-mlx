# SAM3-Tracker-MLX
This code is to run the [SAM3 Tracker](https://huggingface.co/docs/transformers/model_doc/sam3_tracker) model faster on macOS devices using MLX.
We improved the [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) SAM3 Tracker code so that point, box, and mask input prompts are added. If we use the same image resizing and nomalization functions as [PyTorch Transformers implementation](https://github.com/huggingface/transformers), the result mask, iou_score, and object_score_logits are exactly the same between our MLX implementation and the Pytorch Transformers implementation. 

```bash
git clone https://github.com/ryouchinsa/sam3-tracker-mlx.git
cd sam3-tracker-mlx

conda create -n sam3 python=3.12
conda activate sam3
pip install opencv-python mlx pillow requests huggingface_hub safetensors

python test_mlx_tracker.py
```

```bash
pip install torch==2.10.0 torchvision
pip install transformers

test_transformers_tracker.py
```
