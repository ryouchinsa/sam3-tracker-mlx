## SAM3-Tracker-MLX
This code is to run the [SAM3 Tracker](https://huggingface.co/facebook/sam3) model faster on macOS devices using MLX.

We improved the [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) SAM3 Tracker code so that point, box, and mask input prompts work correctly. If we use the same image resizing and nomalization functions as [PyTorch Transformers](https://github.com/huggingface/transformers), the result mask, iou_score, and object_score_logits are exactly the same between our MLX implementation and the Pytorch Transformers implementation. 

Install and run the SAM3 Tracker model using MLX.

```bash
git clone https://github.com/ryouchinsa/sam3-tracker-mlx.git
cd sam3-tracker-mlx

conda create -n sam3 python=3.12
conda activate sam3
pip install opencv-python mlx pillow requests huggingface_hub safetensors

python test_mlx_tracker.py
```

Using the same image resizing and nomalization functions, compare the result mask, iou_score, and object_score_logits between our MLX implementation and the Transformers implementation.

```bash
pip install torch==2.10.0 torchvision
pip install transformers

python test_mlx_tracker.py --preprocess_like_torch
python test_transformers_tracker.py
```

MLX result on our Apple M1 device.

```bash
# load_model: 1019.15 ms
# feature_maps: 5849.87 ms
# pred_masks
(1, 1, 1, 288, 288)
array([[[[[-13.5184, -12.7845, -12.2627, ..., -12.71, -11.5591, -11.3816],
          [-12.9192, -15.5912, -15.5127, ..., -14.1778, -14.4347, -11.439],
          [-15.5103, -13.8262, -15.3593, ..., -15.0312, -14.0051, -14.776],
          ...,
          [-12.6695, -17.5009, -15.1329, ..., -11.4361, -10.9833, -6.98978],
          [-14.5389, -13.0723, -16.0342, ..., -3.04374, -8.40482, -8.00219],
          [-10.421, -13.6726, -10.0483, ..., -8.5702, -5.33487, -7.004]]]]], dtype=float32)
# iou_scores
(1, 1, 1)
array([[[0.618768]]], dtype=float32)
# object_score_logits
(1, 1, 1)
array([[[21.2302]]], dtype=float32)
# predict: 71.85 ms
```

Transformers result on our Apple M1 device.

```bash
# load_model: 2586.00 ms
# processor: 58.41 ms
# pred_masks
torch.Size([1, 1, 1, 288, 288])
tensor([[[[[-13.5184, -12.7845, -12.2627,  ..., -12.7099, -11.5591,
            -11.3816],
           [-12.9191, -15.5912, -15.5127,  ..., -14.1778, -14.4347,
            -11.4390],
           [-15.5103, -13.8262, -15.3593,  ..., -15.0312, -14.0051,
            -14.7760],
           ...,
           [-12.6695, -17.5009, -15.1329,  ..., -11.4362, -10.9834,
             -6.9898],
           [-14.5389, -13.0723, -16.0342,  ...,  -3.0438,  -8.4049,
             -8.0022],
           [-10.4210, -13.6725, -10.0482,  ...,  -8.5702,  -5.3349,
             -7.0040]]]]])
# iou_scores
torch.Size([1, 1, 1])
tensor([[[0.6188]]])
# object_score_logits
torch.Size([1, 1, 1])
tensor([[[21.2302]]])
# predict: 13674.61 ms
```
