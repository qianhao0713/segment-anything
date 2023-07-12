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
from segment_anything.trt_utils import inference as trt_infer
import os,json,time
from pointcloud_cluster_cpp.lib import pointcloud_cluster

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
    mask_image = mask[:,0,:,:].reshape(h, w, 1).cpu() * color.reshape(1, 1, -1)
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

def process_data(data, pred_iou_thresh = 0.88, stability_score_thresh = 0.95):
    pred_iou_thresh = pred_iou_thresh
    mask_threshold = 0.0
    stability_score_thresh = stability_score_thresh
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
    # data.filter(keep_mask)

    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])

def lidar_coords2box(coords):
    x1=np.min(coords[:, :, 0], axis=1)
    x2=np.max(coords[:, :, 0], axis=1)
    y1=np.min(coords[:, :, 1], axis=1)
    y2=np.max(coords[:, :, 1], axis=1)
    box=np.hstack([x1, y1, x2, y2])
    return box

def gen_background_coord(lidar_box, r=1.5, img_size=[1920, 1080]):
    center_x = (lidar_box[:,0]+lidar_box[:,2])/2
    center_y = (lidar_box[:,1]+lidar_box[:,3])/2
    xl=np.maximum(center_x + r * (lidar_box[:, 0] - center_x), 0)
    xr=np.minimum(center_x + r * (lidar_box[:, 2] - center_x), img_size[1])
    yl=np.maximum(center_y + r * (lidar_box[:, 1] - center_y), 0)
    yr=np.minimum(center_y + r * (lidar_box[:, 3] - center_y), img_size[0])
    p1 = np.vstack([xl, yl]).T
    p2 = np.vstack([xl, center_y]).T
    p3 = np.vstack([xl, yr]).T
    p4 = np.vstack([center_x, yl]).T
    p5 = np.vstack([center_x, center_y]).T
    p6 = np.vstack([center_x, yr]).T
    p7 = np.vstack([xr, yl]).T
    p8 = np.vstack([xr, center_y]).T
    p9 = np.vstack([xr, yr]).T
    back_points = np.stack([p1, p2, p3, p4, p5, p6, p7, p8, p9], axis=1)
    return back_points


def get_iou_argmax(bbox1, bbox2):
    b1, _ = bbox1.shape
    b2, n2, _ = bbox2.shape
    bbox1 = torch.tile(bbox1[:,None,:], [1, n2, 1])
    x1_max = torch.maximum(bbox1[:,:,0], bbox2[:,:,0])
    y1_max = torch.maximum(bbox1[:,:,1], bbox2[:,:,1])
    x2_min = torch.minimum(bbox1[:,:,2], bbox2[:,:,2])
    y2_min = torch.minimum(bbox1[:,:,3], bbox2[:,:,3])
    inter_area = torch.maximum(x2_min-x1_max, torch.tensor(0)) * torch.maximum(y2_min-y1_max, torch.tensor(0))
    outer_area = (bbox1[:,:,2]-bbox1[:,:,0]) * (bbox1[:,:,3]-bbox1[:,:,1]) + (bbox2[:,:,2]-bbox2[:,:,0]) * (bbox2[:,:,3]-bbox2[:,:,1]) - inter_area
    iou = inter_area / outer_area
    return torch.argmax(iou, axis=1)


class LidarParam:
    def __init__(self) -> None:
        self.camera_matrix = np.array([1358.080518, 0.0, 987.462437,
                              0.0, 1359.770396, 585.756872,
                              0.0, 0.0, 1.0]).reshape((3, 3))

        self.transform = np.array([
            0.02158791, -0.99976086, 0.00349065, -0.24312639,
            -0.01109192, -0.00373076, -0.99993152, -0.22865444,
            0.99970542, 0.02154772, -0.01116981, -0.37689865
        ], dtype=np.float32).reshape(3, 4)

        self.distortion = np.array([[-0.406858, 0.134080, 0.000104, 0.001794, 0.0]], dtype=np.float32)

        # 获取更新后的 旋转和平移
        self.rMat = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        self.tVec = np.array([0, 0, 0], dtype=np.float32).reshape(1, 3)


class SAMTRT(object):
    def __init__(self, device, conf_path=None, **kwargs):
        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device.isdigit():
            self.device = torch.device('cuda', int(device))
        else:
            self.device = torch.device(device)
        self.conf_path = '%s/samtrt_conf.json' % os.path.dirname(__file__)
        trt_conf = {}
        if conf_path is not None:
            self.conf_path = "%s/%s" % (os.path.dirname(__file__), conf_path)
        with open(self.conf_path) as conf_file:
            trt_conf = json.load(conf_file)
        root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.encoder_trt = "%s/%s" % (root_dir, trt_conf.get('encoder_trt'))
        self.decoder_trt = "%s/%s" % (root_dir, trt_conf.get('decoder_trt'))
        self.origin_image_shape = trt_conf.get('origin_image_shape')
        assert self.encoder_trt is not None and self.decoder_trt is not None, "tensorRT engine file not given"
        assert self.origin_image_shape is not None, "origin_image_shape not given"
        self.generate_masks = trt_conf.get('generate_mask', True)
        self.point_per_side = trt_conf.get('point_per_side', 32)
        self.points_per_batch = trt_conf.get('point_per_batch', 64)
        self.partial_width = trt_conf.get('partial_width', 1.0)
        self.partial_height = trt_conf.get('partial_height', 1.0)
        self.print_infer_delay = trt_conf.get('print_infer_delay', False)
        self.image_size = 1024
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(self.device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(self.device)

        self.use_trt = trt_conf.get('use_trt', True)
        self.use_lidar = trt_conf.get('use_lidar', False)
        if "use_trt" in kwargs:
            self.use_trt = kwargs["use_trt"]
        if "use_lidar" in kwargs:
            self.use_lidar = kwargs["use_lidar"]
        if self.use_lidar:
            self.pointcloud_cluster_tool = pointcloud_cluster.PyPointCloud()
            self.max_point = trt_conf.get('max_point', 40)
            self.cluster_mode = trt_conf.get('cluster_mode', False)
            if self.cluster_mode:
                self.pred_iou_thresh = 0
                self.stability_score_thresh = 0
            else:
                self.pred_iou_thresh = 0.88
                self.stability_score_thresh = 0.95
            self.lidar_param = LidarParam()

            if self.cluster_mode:
                self.coords_input = torch.zeros([self.points_per_batch, self.max_point, 2], dtype=torch.float32, device=self.device)
                self.labels_input = torch.zeros([self.points_per_batch, self.max_point], dtype=torch.float32, device=self.device)
            else:
                self.coords_input = torch.zeros([1024, 1, 2],  dtype=torch.float32, device=self.device)
                self.labels_input = torch.zeros([1024, 1], dtype=torch.float32, device=self.device)
            self.mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self.device)
            self.has_mask_input = torch.zeros(1, dtype=torch.float32, device=self.device)
            self.orig_im_size = torch.as_tensor(self.origin_image_shape, device=self.device)

    def load_sam_trt(self):
        dynamic_shape = {}
        dynamic_shape_value = {}
        dynamic_shape_value['orig_im_size'] = self.origin_image_shape
        if not self.generate_masks:
            dynamic_shape['point_coords'] = [1, 2, 2]
            dynamic_shape['point_labels'] = [1, 2]
        elif self.use_lidar:
            dynamic_shape['point_coords'] = [self.points_per_batch, self.max_point, 2]
            dynamic_shape['point_labels'] = [self.points_per_batch, self.max_point]
        else:
            dynamic_shape['point_coords'] = [self.points_per_batch, 1, 2]
            dynamic_shape['point_labels'] = [self.points_per_batch, 1]
        self.vit_embedding_engine = trt_infer.TRTInference(trt_engine_path=self.encoder_trt, is_torch_infer=True, device=self.device)
        self.mask_decoder_engine = trt_infer.TRTInference(trt_engine_path=self.decoder_trt, dynamic_shape=dynamic_shape, dynamic_shape_value=dynamic_shape_value, is_torch_infer=True, device=self.device)

    def load_sam_base(self):
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        checkpoint = "%s/../weights/sam_vit_l_0b3195.pth" % cur_dir
        model_type = "vit_l"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint).to(self.device)

    def load_sam(self):
        if self.use_trt:
            self.load_sam_trt()
        else:
            self.load_sam_base()

    def _pre_process_image(self, image):
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

    def _pre_process_points(self, points):
        filter = np.all(~np.isnan(points), axis=1)
        points = points[filter, :]
        angle = np.angle(points[:, 0] + 1j * points[:, 1], deg=True) * 5 + 900
        points[:, 3] = angle
        points = points[points[:, 0] > 0]
        return points

    def _project_by_cluster(self, points):
        labels = points[:,5].astype(int)
        coords = []
        for i in range(labels.max() + 1):
            cluster = points[labels == i]
            n_cluster_point = cluster.shape[0]
            if n_cluster_point > self.max_point:
                shuffle = np.random.randint(0, n_cluster_point, size=self.max_point)
                cluster = cluster[shuffle]
            cluster = cluster[:, :3].astype(np.float32)
            tmp_point_cloud = np.hstack((cluster, np.ones([len(cluster), 1])))
            cluster = np.dot(tmp_point_cloud, self.lidar_param.transform.T)
            reTransform = cv2.projectPoints(cluster, self.lidar_param.rMat, self.lidar_param.tVec, self.lidar_param.camera_matrix, self.lidar_param.distortion)
            coord = reTransform[0][:, 0].astype(int)
            filter = np.where((coord[:, 0] < self.origin_image_shape[1]) & (coord[:, 1] < self.origin_image_shape[0]) & (coord[:, 0] >= 0) & (coord[:, 1] >= 0))
            coord = coord[filter]
            if coord.shape[0] == 0:
                continue
            coords.append(coord)
        return coords

    def prepare_image(self, ori_image):
        image = self._pre_process_image(ori_image)
        self.transf = ResizeLongestSide(self.image_size)
        return image

    def infer_vit_embedding(self, ort_inputs1, infer_async=False):
        if self.use_trt:
            if infer_async:
                self.vit_embedding_engine.torch_inference(ort_inputs1, infer_async)
                return
            else:
                return self.vit_embedding_engine.torch_inference(ort_inputs1)[0]
        else:
            return self.sam.image_encoder(ort_inputs1["image"])

    def get_async_vit_result(self):
        return self.vit_embedding_engine.torch_inference_async_result()[0]

    def infer_mask(self, ort_inputs2):
        if self.use_trt:
            iou_token_out, iou_preds, masks = self.mask_decoder_engine.torch_inference(ort_inputs2)
        else:
            torch.cuda.synchronize()
            t1=time.time()
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(ort_inputs2["point_coords"], ort_inputs2["point_labels"]),
                boxes=None,
                masks=None
            )
            low_res_masks, iou_preds, iou_token_out = self.sam.mask_decoder(
                image_embeddings=ort_inputs2["image_embeddings"],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            torch.cuda.synchronize()
            t3=time.time()
            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=[self.image_size, self.image_size],
                original_size=self.origin_image_shape,
            )
            torch.cuda.synchronize()
            t4=time.time()
            print("prompt_encoder: %2.3f, mask_decoder: %2.3f, postprocess: %2.3f" % (t2-t1, t3-t2, t4-t3) )
            best_idx = torch.argmax(iou_preds, dim=1)
            masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
            iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)
        torch.cuda.synchronize()
        return iou_preds.detach(), masks.detach(), iou_token_out.detach()


    def infer_lidar_points(self, image, points):
        t0=time.time()
        ort_inputs1 = {
            "image": image
        }
        self.infer_vit_embedding(ort_inputs1, infer_async=True)

        t0_0=time.time()
        points = self._pre_process_points(points)

        points = self.pointcloud_cluster_tool.execute_cluster(points)
        points = points[points[:,5]>=0]
        coords = self._project_by_cluster(points)
        origin_coord = np.copy(coords)

        t0_1=time.time()

        n_classes = len(coords)
        if self.cluster_mode:
            coord_arr = np.zeros([n_classes, self.max_point, 2])
            label_arr = np.zeros([n_classes, self.max_point])
            lidar_box = np.zeros([n_classes, 4])
            for i in range(n_classes):
                coord = coords[i]
                lidar_box[i, 0] = np.min(coord[:, 0], axis=0)
                lidar_box[i, 1] = np.min(coord[:, 1], axis=0)
                lidar_box[i, 2] = np.max(coord[:, 0], axis=0)
                lidar_box[i, 3] = np.max(coord[:, 1], axis=0)
            for i in range(n_classes):
                coord = coords[i]
                coord_num = coord.shape[0]
                coord_arr[i, :coord_num, :] = coord
                label_arr[i, :coord_num] = 1
                label_arr[i, coord_num:] = -1

            self.coords_input[:n_classes] = torch.from_numpy(coord_arr)
            self.labels_input[:n_classes] = torch.from_numpy(label_arr)
            coords_input = self.coords_input[:n_classes]
            labels_input = self.labels_input[:n_classes]
            lidar_box = torch.from_numpy(lidar_box).to(self.device)
        else:
            coord = np.concatenate(coords, axis=0)
            n_coord = coord.shape[0]
            self.coords_input[:n_coord,...] = torch.from_numpy(coord[:,None,:])
            self.labels_input[:n_coord,...] = 1
            coords_input = self.coords_input[:n_coord]
            labels_input = self.labels_input[:n_coord]
        t0_2=time.time()
        vit_embedding = self.get_async_vit_result()
        t1=time.time()
        ort_inputs2 = {
            "image_embeddings": vit_embedding,
            "mask_input": self.mask_input,
            "has_mask_input": self.has_mask_input,
            "orig_im_size": self.orig_im_size
        }
        res = []
        mask_data = MaskData()
        for (coord_input, label_input) in batch_iterator(self.points_per_batch, coords_input, labels_input):
            coord_input = self.transf.apply_coords(coord_input, self.origin_image_shape)
            ort_inputs2["point_coords"] = coord_input
            ort_inputs2["point_labels"] = label_input
            iou_preds, masks, iou_token_out = self.infer_mask(ort_inputs2)
            if self.cluster_mode:
                sam_box = batched_mask_to_box(masks>0)
                args=get_iou_argmax(lidar_box, sam_box)
                for i in range(args.shape[0]):
                    masks[i, 0, ...] = masks[i, args[i], ...]
                masks = masks[:, 0, ...][:, None, ...]
                iou_preds = iou_preds[:, 0, ...][:, None, ...]
            batch_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                points=torch.as_tensor(coord_input.repeat([masks.shape[1],1,1])),
                iou_token_out=iou_token_out.flatten(0, 1)
            )
            process_data(batch_data, self.pred_iou_thresh, self.stability_score_thresh)
            mask_data.cat(batch_data)
        keep_by_nms = batched_nms(
            mask_data["boxes"].float(),
            mask_data["iou_preds"],
            torch.zeros_like(mask_data["boxes"][:, 0]),  # categories
            iou_threshold=0.77,
        )
        t2=time.time()
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
                "iou_token_out": mask_data["iou_token_out"][idx],
            }
            res.append(ann)
        t3=time.time()
        if self.print_infer_delay:
            print("vit_infer: %2.3f ms, lidar_processing: %2.3f ms, input_h2d: %2.3f ms, mask_infer: %2.3f ms, postprocess: %2.3f ms, all: %2.3f ms" % ((t1-t0) * 1000, (t0_1-t0_0)*1000, (t0_2-t0_1)*1000, (t2-t1) * 1000, (t3-t2)*1000, (t3-t0)*1000))
        return origin_coord, res


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
            coord_input = self.transf.apply_coords(coord_input, image.shape[-2:])
            ort_inputs2["point_coords"] = coord_input
            iou_preds, masks, iou_token_out = self.infer_mask(ort_inputs2)
            batch_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                iou_token_out=iou_token_out.flatten(0, 1),
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
                "iou_token_out": mask_data["iou_token_out"][idx],
            }
            res.append(ann)
        post_process_time2 = time.time()-t4
        if self.print_infer_delay:
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
        t0=time.time()
        vit_embedding = self.infer_vit_embedding(ort_inputs1)
        ort_inputs2["image_embeddings"] = vit_embedding
        t1=time.time()
        iou_preds, masks = self.infer_mask(ort_inputs2)
        t2=time.time()
        vit_infer_time = t1-t0
        mask_infer_time = t2-t1
        if self.print_infer_delay:
            print("vit_infer: %2.3f, mask_infer: %2.3f" % (vit_infer_time, mask_infer_time))
        return iou_preds, masks

    def show_result(self, masks=None, image=None, point=None, label=None, anns=None, out_path=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        if self.generate_masks or self.use_lidar:
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for i, ann in enumerate(sorted_anns):
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
                show_box(ann['bbox'], ax, i)
            ax.imshow(img)
            plt.axis('on')
            plt.savefig(out_path)
        else:
            masks = masks > 0.0
            show_mask(masks, plt.gca())
            show_points(point, label, plt.gca())
            plt.axis('on')
            plt.savefig(out_path)
