## SAM3-Tracker-MLX
This code is to run the [SAM3 Tracker](https://huggingface.co/facebook/sam3) model faster on macOS devices using MLX.

We improved the [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) SAM3 Tracker code so that point, box, and mask input prompts work correctly. If we use the same image resizing and nomalization functions as [PyTorch Transformers](https://github.com/huggingface/transformers), the result mask, iou_score, and object_score_logits are exactly the same between our MLX implementation and the Pytorch Transformers. 

Install and run the SAM3 Tracker model using MLX.

```bash
git clone https://github.com/ryouchinsa/sam3-tracker-mlx.git
cd sam3-tracker-mlx

conda create -n sam3 python=3.12
conda activate sam3
pip install opencv-python mlx pillow requests huggingface_hub safetensors

python test_mlx_tracker.py
```

Using the same image resizing and nomalization functions, compare the result mask, iou_score, and object_score_logits between our MLX implementation and the PyTorch Transformers.

```bash
pip install torch==2.10.0 torchvision
pip install transformers

python test_mlx_tracker.py --preprocess_like_torch
python test_transformers_tracker.py
```

### MLX result on our Apple M1 device.

```bash
# load_model: 853.90 ms
# feature_maps: 4588.41 ms
# pred_masks
(1, 1, 1, 288, 288)
array([[[[[-13.5089, -12.762, -12.287, ..., -12.8383, -11.6467, -11.4884],
          [-12.8803, -15.6617, -15.6493, ..., -14.2366, -14.5536, -11.5195],
          [-15.5604, -13.8372, -15.4348, ..., -15.1119, -14.0629, -14.8326],
          ...,
          [-12.6745, -17.5274, -15.2105, ..., -11.8795, -11.4878, -7.16195],
          [-14.534, -13.0463, -16.0124, ..., -3.00889, -8.73899, -8.19234],
          [-10.4175, -13.6745, -10.0635, ..., -8.88442, -5.40519, -7.22509]]]]], dtype=float32)
# iou_scores
(1, 1, 1)
array([[[0.621079]]], dtype=float32)
# object_score_logits
(1, 1, 1)
array([[[21.1843]]], dtype=float32)
# predict: 67.36 ms

```

### PyTorch Transformers result on our Apple M1 device.

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
