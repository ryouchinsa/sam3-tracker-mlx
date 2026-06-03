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

MLX result.

```bash
# load_model: 1063.03 ms
# feature_maps: 6445.35 ms
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
# predict: 91.02 ms
# pred_masks
(1, 1, 1, 288, 288)
array([[[[[-11.5506, -12.3197, -10.8077, ..., -12.62, -10.5808, -10.7619],
          [-11.4273, -13.2322, -9.06491, ..., -12.3486, -10.8494, -10.6088],
          [-12.8223, -14.724, -13.2716, ..., -14.075, -12.463, -13.823],
          ...,
          [-9.39367, -12.8007, -7.43116, ..., -9.76384, -4.50621, -7.5954],
          [-9.95378, -12.4816, -11.9414, ..., -6.26456, -7.99747, -7.08613],
          [-8.87647, -9.60523, -7.85607, ..., -6.44335, -5.21303, -6.35633]]]]], dtype=float32)
# iou_scores
(1, 1, 1)
array([[[0.956187]]], dtype=float32)
# object_score_logits
(1, 1, 1)
array([[[27.6314]]], dtype=float32)
# predict: 24.70 ms
```

Transformers result.

```bash
# load_model: 2586.00 ms
# processor: 58.41 ms
/Users/ryo/Downloads/sam3-mlx/test_transformers_tracker.py:22: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  pred_mask = np.array(pred_mask)
/Users/ryo/Downloads/sam3-mlx/test_transformers_tracker.py:29: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  object_score_logit = 1.0 / (1.0 + np.exp(-object_score_logit))
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
# processor: 1.19 ms
# pred_masks
torch.Size([1, 1, 1, 288, 288])
tensor([[[[[-11.5506, -12.3197, -10.8077,  ..., -12.6199, -10.5808,
            -10.7619],
           [-11.4272, -13.2322,  -9.0649,  ..., -12.3486, -10.8493,
            -10.6088],
           [-12.8223, -14.7240, -13.2716,  ..., -14.0750, -12.4630,
            -13.8230],
           ...,
           [ -9.3937, -12.8007,  -7.4311,  ...,  -9.7639,  -4.5062,
             -7.5954],
           [ -9.9538, -12.4816, -11.9415,  ...,  -6.2646,  -7.9975,
             -7.0861],
           [ -8.8765,  -9.6052,  -7.8561,  ...,  -6.4433,  -5.2130,
             -6.3563]]]]])
# iou_scores
torch.Size([1, 1, 1])
tensor([[[0.9562]]])
# object_score_logits
torch.Size([1, 1, 1])
tensor([[[27.6314]]])
# predict: 47.18 ms
```
