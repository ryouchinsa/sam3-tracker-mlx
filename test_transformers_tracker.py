from transformers import Sam3TrackerProcessor, Sam3TrackerModel
import torch
from PIL import Image
import requests
import numpy as np
import cv2
import time

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
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    input_points = [[[[1250,530]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
    input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)
    inputs = processor(
        images=image, 
        input_points=input_points, 
        input_labels=input_labels, 
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    mask_input = outputs.pred_masks[:, :, 0]
    input_points = [[[[1250,530], [1040,590]]]]
    input_labels = [[[1, 1]]]
    inputs = processor(
        input_points=input_points, 
        input_labels=input_labels, 
        original_sizes=inputs["original_sizes"],
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            input_masks=mask_input, 
            image_embeddings=outputs.image_embeddings,
            multimask_output=False
        )
    visualize_results([image], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask_2nd")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_box():
    start = time.perf_counter()
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    input_boxes = [[[1025, 205, 1075, 330]]]
    inputs = processor(
        images=image, 
        input_boxes=input_boxes, 
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_points_and_box():
    start = time.perf_counter()
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    input_points = [[[[1040,590]]]]
    input_labels = [[[1]]]
    input_boxes = [[[1010, 460, 1420, 750]]]
    inputs = processor(
        images=image, 
        input_points=input_points, 
        input_labels=input_labels,
        input_boxes=input_boxes, 
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_points_multiple_objects():
    start = time.perf_counter()
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    input_points = [[[[1050, 270]], [[930, 140]]]]
    input_labels = [[[1], [1]]]
    inputs = processor(
        images=image, 
        input_points=input_points, 
        input_labels=input_labels,
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_batch_images():
    start = time.perf_counter()
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    image2 = Image.open("jesse-hammer-4fWuS52jENk-unsplash.jpg")
    input_points = [[[[1050, 270]]], [[[520, 730]]]]
    input_labels = [[[1]], [[1]]]
    inputs = processor(
        images=[image, image2], 
        input_points=input_points, 
        input_labels=input_labels,
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image, image2], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_batched_objects_per_image():
    start = time.perf_counter()
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    image2 = Image.open("jesse-hammer-4fWuS52jENk-unsplash.jpg")
    input_points = [
        [[[1050, 270]], [[930, 140]]],
        [[[520, 730]]]
    ]
    input_labels = [
        [[1], [1]],
        [[1]]
    ]
    inputs = processor(
        images=[image, image2], 
        input_points=input_points, 
        input_labels=input_labels,
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image, image2], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_batched_images_and_objects():
    start = time.perf_counter()
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    image2 = Image.open("jesse-hammer-4fWuS52jENk-unsplash.jpg")
    input_points = [
        [[[1040, 590]], [[1050, 270]]],
        [[[280, 400]], [[290, 950], [620, 600]]]
    ]
    input_labels = [
        [[1], [1]],
        [[1], [1, 1]]
    ]
    inputs = processor(
        images=[image, image2], 
        input_points=input_points, 
        input_labels=input_labels,
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image, image2], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

def test_batched_boxes():
    start = time.perf_counter()
    model = Sam3TrackerModel.from_pretrained("facebook/sam3")
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"# load_model: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    image = Image.open("david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg")
    image2 = Image.open("jesse-hammer-4fWuS52jENk-unsplash.jpg")
    input_boxes = [
        [[1025, 205, 1075, 330], [1010, 460, 1420, 750]],
        [[170, 500, 720, 1110], [200, 340, 420, 480]]
    ]
    inputs = processor(
        images=[image, image2], 
        input_boxes=input_boxes, 
        return_tensors="pt"
    )
    print(f"# processor: {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            **inputs, 
            multimask_output=False
        )
    visualize_results([image, image2], outputs.pred_masks, outputs.iou_scores, outputs.object_score_logits, "pred_mask")
    print(f"# predict: {(time.perf_counter() - start) * 1000:.2f} ms")

if __name__ == "__main__":
    test_points_using_previous_mask()
    # test_box()
    # test_points_and_box()
    # test_points_multiple_objects()
    # test_batch_images()
    # test_batched_objects_per_image()
    # test_batched_images_and_objects()
    # test_batched_boxes()



























