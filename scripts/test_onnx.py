import torch
import torch.nn as nn
from torch.nn import functional as F
from segment_anything.modeling import Sam
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from typing import Tuple
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import onnxruntime

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

# @torch.no_grad()
def pre_processing(image: np.ndarray, target_length: int, device,pixel_mean,pixel_std,img_size):
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    input_image = np.array(resize(to_pil_image(image), target_size))
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize colors
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std

    # Pad
    h, w = input_image_torch.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask[:,3,:,:].reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def onnx_model_example():
    import os
    ort_session_embedding = onnxruntime.InferenceSession('./weights/sam_image_encoder.onnx',providers=['CPUExecutionProvider'])
    ort_session_sam = onnxruntime.InferenceSession('./weights/sam_mask_decoder.onnx',providers=['CPUExecutionProvider'])

    image = cv2.imread('notebooks/images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = image.copy()
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    target_length = image_size
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    device = "cpu"
    inputs = pre_processing(image, target_length, device,pixel_mean,pixel_std,image_size)
    ort_inputs = {
    "image": inputs.cpu().numpy()
    }
    image_embeddings = ort_session_embedding.run(None, ort_inputs)[0]

    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = "weights/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)


    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    from segment_anything.utils.transforms import ResizeLongestSide
    transf = ResizeLongestSide(image_size)
    onnx_coord = transf.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    ort_inputs = {
        "image_embeddings": image_embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int32)
    }

    masks, iou_preds, _ = ort_session_sam.run(None, ort_inputs)
    print(iou_preds)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    onnx_model_path = "weights/sam_onnx_example.onnx"

    onnx_model = SamOnnxModel(sam, return_single_mask=False)
    # print(masks.shape)
    # # masks = onnx_model.mask_postprocessing(torch.as_tensor(masks), torch.as_tensor(image.shape[:2]))

    # masks = masks > 0.0
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_mask(masks, plt.gca())
    # # show_box(input_box, plt.gca())
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.savefig('demo3.png')


with torch.no_grad():
    onnx_model_example()
