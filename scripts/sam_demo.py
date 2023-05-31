import torch
from torch.nn import functional as F
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image
from typing import Tuple
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import build_point_grid, batch_iterator, MaskData, calculate_stability_score, batched_mask_to_box, area_from_mask, box_xyxy_to_xywh
from segment_anything import sam_model_registry
from torchvision.ops.boxes import batched_nms
import cv2
import matplotlib.pyplot as plt
from trt_utils import inference as trt_infer
import argparse
import os, sys, time
import pycuda.driver as cuda
import pycuda.autoinit

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask[:,0,:,:].reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
def show_box(box, ax, color_index):
    colors = ['r', 'g', 'y', 'b', 'c']
    color_index = color_index % len(colors)
    color = colors[color_index]
    x0, y0 = box[0], box[1]
    w, h = box[2], box[3]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))

def process_data(data):
    pred_iou_thresh = 0.88
    mask_threshold = 0.0
    stability_score_thresh = 0.95
    stability_score_offset = 1.0
    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)
    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
        data["masks"], mask_threshold, stability_score_offset
    )
    keep_mask = data["stability_score"] >= stability_score_thresh
    data.filter(keep_mask)
    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_coord_input', action='store_true')
    parser.add_argument('--exclude_postprocess', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--encoder_trt', type=str, default='weights/sam_image_encoder.trt', help='sam image encoder trt path')
    parser.add_argument('--decoder_trt', type=str, default='weights/sam_single_mask_mask_decoder_fold.trt', help='sam mask decoder trt path')
    # parser.add_argument('--decoder_trt', type=str, default='weights/sam_mask_decoder_fold.trt', help='sam mask decoder trt path')
    parser.add_argument('--points_per_side', type=int, default=32, help='point input num in one-side of a picture for mask-generation, total num is n_points * n_points')
    parser.add_argument('--points_per_batch', type=int, default=64, help='point input in one batch for mask-generation')
    parser.add_argument('--box_nms_thresh', type=float, default=0.77)
    args = parser.parse_args()
    return args

class SAMTRT(object):
    def __init__(self, args, use_trt=True):
        if args.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', int(args.device))
        self.encoder_trt = args.encoder_trt
        self.decoder_trt = args.decoder_trt
        self.generate_masks = False if args.single_coord_input else True
        self.need_postprocess = False if args.exclude_postprocess else True
        self.origin_image_shape = [1200, 1800]
        # self.origin_image_shape = [534, 800]
        self.image_size = 1024
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375]
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(self.device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(self.device)
        self.point_per_side = 32 if not args.points_per_side else args.points_per_side 
        self.points_per_batch = 64 if not args.points_per_batch else args.points_per_batch
        self.partial_width = 1.0
        self.partial_height = -0.5
        self.use_trt=use_trt  

    def load_sam_trt(self):
        dynamic_shape = {}
        dynamic_shape_value = {}
        dynamic_shape_value['orig_im_size'] = self.origin_image_shape
        if not self.generate_masks:
            dynamic_shape['point_coords'] = [1, 2, 2]
            dynamic_shape['point_labels'] = [1, 2]
        else:
            dynamic_shape['point_coords'] = [self.points_per_batch, 1, 2]
            dynamic_shape['point_labels'] = [self.points_per_batch, 1]
        self.vit_embedding_engine = trt_infer.TRTInference(trt_engine_path=self.encoder_trt, is_torch_infer=True, device=self.device)
        self.mask_decoder_engine = trt_infer.TRTInference(trt_engine_path=self.decoder_trt, dynamic_shape=dynamic_shape, dynamic_shape_value=dynamic_shape_value, is_torch_infer=True, device=self.device)

    def load_sam_base(self):
        checkpoint = "weights/sam_vit_l_0b3195.pth"
        model_type = "vit_l"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint).to(self.device)
        
    def load_sam(self):
        if self.use_trt:
            self.load_sam_trt()
        else:
            self.load_sam_base()
        
    def _pre_process(self, image):
        target_size = get_preprocess_shape(image.shape[0], image.shape[1], self.image_size)
        input_image = np.array(resize(to_pil_image(image), target_size))
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Normalize colors
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = input_image_torch.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
        return input_image_torch
    
    def prepare_image(self, image_path):
        ori_image = cv2.imread(image_path)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image = self._pre_process(ori_image)
        self.transf = ResizeLongestSide(self.image_size)
        return ori_image, image
    
    def infer_vit_embedding(self, ort_inputs1):
        if self.use_trt:
            return self.vit_embedding_engine.torch_inference(ort_inputs1)[0]
        else:
            return self.sam.image_encoder(ort_inputs1["image"])
        
    def infer_mask(self, ort_inputs2):
        if self.use_trt:
            _, iou_preds, masks = self.mask_decoder_engine.torch_inference(ort_inputs2)
        else:
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(ort_inputs2["point_coords"], ort_inputs2["point_labels"]),
                boxes=None,
                masks=None
            )
            low_res_masks, iou_preds = self.sam.mask_decoder(
                image_embeddings=ort_inputs2["image_embeddings"],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=[1024, 1024],
                original_size=self.origin_image_shape,
            )
            best_idx = torch.argmax(iou_preds, dim=1)
            masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
            iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)
            masks = masks.detach()
            
        torch.cuda.synchronize()
        return iou_preds.detach(), masks.detach()
        
    def infer_grid_coord(self, image):
        point_grids = build_point_grid(self.point_per_side, partial_y=self.partial_width, partial_x=self.partial_height)
        points_scale = np.array(self.origin_image_shape)[None, :]
        coord_grids = torch.as_tensor((point_grids * points_scale)[:,None,:], dtype=torch.float32, device=self.device)
        label_input = torch.ones([self.points_per_batch, 1], dtype=torch.float32, device=self.device)
        mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self.device)
        has_mask_input = torch.zeros(1, dtype=torch.float32, device=self.device)
        orig_im_size = torch.as_tensor(self.origin_image_shape, device=self.device)
        ort_inputs1 = {
            "image": image
        }
        ort_inputs2 = {
            "point_labels": label_input,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size
        }
        mask_data = MaskData()
        t0=time.time()
        vit_embedding = self.infer_vit_embedding(ort_inputs1)
        ort_inputs2["image_embeddings"] = vit_embedding
        vit_embedding_time = time.time()-t0
        mask_decoder_time = 0
        post_process_time1 = 0
        post_process_time2 = 0
        res = []
        for (coord_input,) in batch_iterator(self.points_per_batch, coord_grids):
            t1 = time.time()
            coord_input = self.transf.apply_coords(coord_input, image.shape[:2])
            ort_inputs2["point_coords"] = coord_input
            iou_preds, masks = self.infer_mask(ort_inputs2)
            batch_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                points=torch.as_tensor(coord_input.repeat([masks.shape[1],1,1])),
            )
            t2=time.time()
            process_data(batch_data)
            mask_data.cat(batch_data)
            t3=time.time()
            mask_decoder_time += t2-t1
            post_process_time1 += t3-t2
        t4 = time.time()
        keep_by_nms = batched_nms(
            mask_data["boxes"].float(),
            mask_data["iou_preds"],
            torch.zeros_like(mask_data["boxes"][:, 0]),  # categories
            iou_threshold=0.77,
        )
        mask_data.filter(keep_by_nms)
        mask_data["segmentations"] = mask_data["masks"]
        mask_data.to_numpy()
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_mask(mask_data["masks"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
            }
            res.append(ann)
        post_process_time2 = time.time()-t4
        print("vit_infer: %2.3f, mask_infer: %2.3f, postprocess: %2.3f" % (vit_embedding_time, mask_decoder_time, post_process_time1 + post_process_time2))
        return res

    def infer_single_coord(self, image, coord, label):
        coord_input = np.concatenate([coord, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        label_input = np.concatenate([label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        coord_input = self.transf.apply_coords(coord_input, self.origin_image_shape).astype(np.float32)
        
        coord_input = torch.as_tensor(coord_input, device=self.device)
        label_input = torch.as_tensor(label_input, device=self.device)

        mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self.device)
        has_mask_input = torch.zeros(1, dtype=torch.float32, device=self.device)
        orig_im_size = torch.as_tensor(self.origin_image_shape, device=self.device)
        ort_inputs1 = {
            "image": image
        }
        ort_inputs2 = {
            "point_coords": coord_input,
            "point_labels": label_input,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size
        }
        vit_embedding = self.infer_vit_embedding(ort_inputs1)
        ort_inputs2["image_embeddings"] = vit_embedding
        iou_preds, masks = self.infer_mask(ort_inputs2)
        return iou_preds, masks
    
    def show_result(self, masks=None, image=None, point=None, label=None, anns=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        if self.generate_masks:
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
            for i, ann in enumerate(sorted_anns):
                show_box(ann['bbox'], ax, i)
            ax.imshow(img)
            plt.axis('on')
            plt.savefig('sam_grid_demo.png')
        else:
            masks = masks > 0.0
            show_mask(masks, plt.gca())
            show_points(point, label, plt.gca())
            plt.axis('on')
            plt.savefig('sam_single_demo.png')

def test_single():
    args = parse_args()
    point = np.array([[500, 375]])
    label = np.array([1])

    print("use sam tensort-rt")
    sam_trt = SAMTRT(args, use_trt=True)
    sam_trt.generate_masks = False
    sam_trt.load_sam()
    origin_image, image = sam_trt.prepare_image('notebooks/images/truck.jpg')

    iou_preds, masks = sam_trt.infer_single_coord(image, point, label)
    
    sam_trt.show_result(masks, origin_image, point, label)

    print("use sam base")
    sam_trt = SAMTRT(args, use_trt=False)
    sam_trt.generate_masks = False
    sam_trt.load_sam()
    origin_image, image = sam_trt.prepare_image('notebooks/images/truck.jpg')

    iou_preds, masks = sam_trt.infer_single_coord(image, point, label)

def test_grids():
    args = parse_args()
    print("use sam tensort-rt")
    sam_trt = SAMTRT(args, use_trt=True)
    sam_trt.load_sam()
    origin_image, image = sam_trt.prepare_image('notebooks/images/truck.jpg')
    res = sam_trt.infer_grid_coord(image)
    sam_trt.show_result(image=origin_image, anns=res)

    print("use sam base")
    sam_trt = SAMTRT(args, use_trt=False) 
    sam_trt.load_sam()
    origin_image, image = sam_trt.prepare_image('notebooks/images/truck.jpg')
    res=sam_trt.infer_grid_coord(image)
        
if __name__ == '__main__':
    test_grids()

    