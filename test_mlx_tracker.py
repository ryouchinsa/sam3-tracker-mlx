import argparse
from PIL import Image
from mlx_vlm_rectlabel.utils import load_model, get_model_path
from mlx_vlm_rectlabel.utils_load import load_model_mlx
from mlx_vlm_rectlabel.models.sam3.generate import Sam3VideoPredictor
from pathlib import Path
import mlx.core as mx
import numpy as np
import cv2
import time

def convert_to_float16():
    model_path = get_model_path("facebook/sam3")
    model = load_model(model_path, convert=True)

def load_model_util():
    use_float32_model = True
    if use_float32_model:
        model_path = get_model_path("facebook/sam3")
        model = load_model(model_path)
    else:
        model_path = Path("sam3_float16")
        model = load_model_mlx(model_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_like_torch", action="store_true")
    args = parser.parse_args()
    video_predictor = Sam3VideoPredictor(model, preprocess_like_torch=args.preprocess_like_torch)
    return video_predictor

def get_images_util(multiple_images=False):
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    images = [image]
    if multiple_images:
        image2 = Image.open("jesse-hammer-4fWuS52jENk-unsplash.jpg")
        images.append(image2)
    original_sizes = [image.size for image in images]
    return images, original_sizes

def visualize_results(
    images: [np.ndarray],
    pred_masks: np.ndarray,
    iou_scores: np.ndarray,
    object_score_logits: np.ndarray,
    output_path: str,
):
    batch_num = pred_masks.shape[0]
    object_num = pred_masks.shape[1]
    for b in range(batch_num):
        image = images[b]
        for o in range(object_num):
            pred_mask = pred_masks[b][o][0]
            pred_mask = np.array(pred_mask)
            pred_mask = cv2.resize(pred_mask, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
            pred_mask = pred_mask > 0.0
            pred_mask = (pred_mask.astype(np.uint8) * 255)
            cv2.imwrite(output_path + "_b" + str(b) + "_o" + str(o) + ".png", pred_mask)
            iou_score = iou_scores[b][o]
            object_score_logit = object_score_logits[b][o]
            object_score_logit = 1.0 / (1.0 + np.exp(-object_score_logit))
    print("# pred_masks")
    print(pred_masks.shape)
    print(pred_masks)
    print("# iou_scores")
    print(iou_scores.shape)
    print(iou_scores)
    print("# object_score_logits")
    print(object_score_logits.shape)
    print(object_score_logits) 

def test_points_using_previous_mask():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util()
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_points = [[[[1250,530]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
    input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_points=input_points, 
        input_labels=input_labels, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_points = [[[[1250,530], [1040,590]]]]
    input_labels = [[[1, 1]]]
    mask_input = pred_masks[:, :, 0]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_points=input_points, 
        input_labels=input_labels, 
        input_masks=mask_input, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask_2nd")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_box():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util()
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_boxes = [[[1025, 205, 1075, 330]]]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_boxes=input_boxes, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_points_and_box():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util()
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_points = [[[[1040,590]]]]
    input_labels = [[[1]]]
    input_boxes = [[[1010, 460, 1420, 750]]]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_points=input_points, 
        input_labels=input_labels, 
        input_boxes=input_boxes, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_points_multiple_objects():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util()
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_points = [[[[1050, 270]], [[930, 140]]]]
    input_labels = [[[1], [1]]]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_points=input_points, 
        input_labels=input_labels, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")
    
def test_batch_images():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util(multiple_images=True)
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_points = [[[[1050, 270]]], [[[520, 730]]]]
    input_labels = [[[1]], [[1]]]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_points=input_points, 
        input_labels=input_labels, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_batched_objects_per_image():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util(multiple_images=True)
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_points = [
        [[[1050, 270]], [[930, 140]]],
        [[[520, 730]]]
    ]
    input_labels = [
        [[1], [1]],
        [[1]]
    ]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_points=input_points, 
        input_labels=input_labels, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_batched_images_and_objects():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util(multiple_images=True)
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_points = [
        [[[1040, 590]], [[1050, 270]]],
        [[[280, 400]], [[290, 950], [620, 600]]]
    ]
    input_labels = [
        [[1], [1]],
        [[1], [1, 1]]
    ]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_points=input_points, 
        input_labels=input_labels, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_batched_boxes():
    start = time.perf_counter()
    video_predictor = load_model_util()
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    images, original_sizes = get_images_util(multiple_images=True)
    feature_maps = video_predictor.get_feature_maps(images)
    print(f"# feature_maps: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    input_boxes = [
        [[1025, 205, 1075, 330], [1010, 460, 1420, 750]], 
        [[170, 500, 720, 1110], [200, 340, 420, 480]]
    ]
    pred_masks, iou_scores, object_score_logits = video_predictor.add_points_boxes(
        feature_maps, 
        original_sizes,
        input_boxes=input_boxes, 
        multimask_output=False
    )
    visualize_results(images, pred_masks, iou_scores, object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

if __name__ == "__main__":
    # convert_to_float16()
    test_points_using_previous_mask()
    # test_box()
    # test_points_and_box()
    # test_points_multiple_objects()
    # test_batch_images()
    # test_batched_objects_per_image()
    # test_batched_images_and_objects()
    # test_batched_boxes()
    






























