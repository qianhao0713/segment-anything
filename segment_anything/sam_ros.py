from abc import ABCMeta, abstractmethod
import os, json
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image
from segment_anything.trt_utils import inference as trt_infer
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import tensorrt as trt
from typing import Tuple
import cv2
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import build_point_grid, batch_iterator, MaskData, calculate_stability_score, batched_mask_to_box, area_from_mask, box_xyxy_to_xywh
from torchvision.ops.boxes import batched_nms
from CUDA_CCL import cuda_ccl


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def resample_coords(coords, max_point, n_resample):
    import random
    coords_resampled = []
    labels = []
    for i, coord in enumerate(coords):
        n_coord = coord.shape[0]
        if n_coord < max_point:
            coords_resampled.append(coord)
            labels.append(i)
        else:
            indexes=list(range(n_coord))
            random.shuffle(indexes)
            duplicates = max_point * n_resample / len(indexes)
            if duplicates > 0:
                indexes = indexes * (int(duplicates) + 1)
            for j in range(n_resample):
                coords_resampled.append(coord[indexes[j*max_point: (j+1)*max_point]])
                labels.append(i)
    return labels, coords_resampled

def process_data(data, cluster_mode=False, use_lidar=False):
    pred_iou_thresh = 0.88
    mask_threshold = 0.0
    stability_score_thresh = 0.95
    stability_score_offset = 1.0
    # Filter by predicted IoU
    if use_lidar:
        stability_score_thresh = 0.5
        if cluster_mode:
            lidar_iou_thresh = 0.5
            keep_mask = data["lidar_iou"] > lidar_iou_thresh
            data.filter(keep_mask)
            pred_iou_thresh = 0.5
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
    # if use_lidar:
    #     mask_label = torch.zeros_like(data["masks"], dtype=torch.int32, device=data["masks"].device)
    #     cuda_ccl.torch_ccl(mask_label, data["masks"], mask_label.shape[1], mask_label.shape[2])
    #     for i in range(mask_label.shape[0]):
    #         mask_label_i=mask_label[i]
    #         mask_i = data["masks"][i]
    #         if mask_i.nonzero().shape[0]==0:
    #             continue
    #         mode_value = torch.mode(mask_label_i[mask_i]).values.item()
    #         #tlabel, tcount = torch.unique(mask_label_i[mask_i], return_counts=True)
    #         #maxlabelcount = torch.argmax(tcount)
    #         #maxlabel = tlabel[maxlabelcount].item()
    #         #mask_i[mask_label_i!=maxlabel] = False
    #         mask_i[mask_label_i!=mode_value] = False
    data["boxes"] = batched_mask_to_box(data["masks"])

def get_lidar_iou(bbox1, bbox2):
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
    return iou

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
        
class LidarParam2:
    def __init__(self) -> None:
        self.camera_matrix = np.array([1400.660521866737, 0, 989.6663020916587,
                      0, 1397.477295771064, 594.9904177802305,
                      0.0, 0.0, 1.0]).reshape((3, 3))
        self.distortion = np.array([[-0.05694747808562394, -0.08329212973455258, -0.0009314071183112955, 0.006712153417379347, 0.2493801178842133]], dtype=np.float32)
        self.rMat = np.array([1.680647483853886, 0.03782614262625567, 0.003707885488685594], dtype=np.float32).reshape(3, 1)
        self.tVec = np.array([-0.3270823619145787, 1.994427053985835, -0.2688515838179673], dtype=np.float32).reshape(1, 3)

class SamRosBase(metaclass=ABCMeta):
    def __init__(self, conf_file, device=0):
        self.device = device
        self._load_conf(conf_file)
        self._load_model()

    @abstractmethod
    def _load_conf(self, conf_file):
        with open(conf_file) as f:
            self.conf = json.load(f)

    @abstractmethod
    def _load_model(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self, *inputs):
        raise NotImplementedError

    def __del__(self):
        if self.model is not None:
            self.model.cfx.pop()

class SamRosVit(SamRosBase):
    def __init__(self, conf_file, device=0):
        super().__init__(conf_file, device)
        self._allocate_buffers()
        pixel_mean_raw = [123.675, 116.28, 103.53]
        pixel_std_raw = [58.395, 57.12, 57.375]
        self.pixel_mean = torch.Tensor(pixel_mean_raw).view(-1, 1, 1).to(self.device)
        self.pixel_std = torch.Tensor(pixel_std_raw).view(-1, 1, 1).to(self.device)

    def _load_conf(self, conf_file):
        super()._load_conf(conf_file)
        self.trt_path = self.conf.get('trt_path')
        self.buffer_size = self.conf.get('buffer_size')

    def _load_model(self):
        self.model = trt_infer.TRTInference(trt_engine_path=self.trt_path, is_torch_infer=True, device=self.device)

    def _allocate_buffers(self):
        drv.init()
        dev = drv.Device(self.device)
        self.ctx_gpu = dev.make_context()
        self.buffer_index = -1
        buffer_shape = [1, 256, 64, 64]
        buffer_dtype = np.float32
        self.buffers = [gpuarray.GPUArray(buffer_shape, dtype=buffer_dtype) for _ in range(self.buffer_size)]
        self.ipc_handles = [drv.mem_get_ipc_handle(self.buffers[i].ptr).hex() for i in range(self.buffer_size)]
        self.check_buffers = [gpuarray.GPUArray([1], dtype=np.int32) for _ in range(self.buffer_size)]
        self.check_handles = [drv.mem_get_ipc_handle(self.check_buffers[i].ptr).hex() for i in range(self.buffer_size)]
        assert len(set(self.ipc_handles)) == self.buffer_size

    def _pre_process_image(self, image):
        image_size = 1024
        target_size = get_preprocess_shape(image.shape[0], image.shape[1], image_size)
        input_image = np.array(resize(to_pil_image(image), target_size))
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # Normalize colors
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = input_image_torch.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
        return input_image_torch

    def infer(self, *inputs):
        origin_image = inputs[0]
        image = self._pre_process_image(origin_image)
        ort_inputs = {
            "image": image
        }
        image_embeddings, inner_state = self.model.torch_inference(ort_inputs)
        return image_embeddings, inner_state


    def store_buffer(self, image_embeddings, check=None):
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        drv.memcpy_dtod(self.buffers[self.buffer_index].ptr, image_embeddings.data_ptr(), trt.volume(image_embeddings.shape)*self.buffers[0].dtype.itemsize)
        if check:
            self.check_buffers[self.buffer_index][0]=check


    def result_info(self):
        return {
            "ipc_handle": self.ipc_handles[self.buffer_index],
            "check_handle": self.check_handles[self.buffer_index],
            "shape": [1, 256, 64, 64],
        }

    def __del__(self):
        self.ctx_gpu.pop()
        self.model.cfx.pop()

class SamRosSeghead(SamRosBase):
    def __init__(self, conf_file, device=0):
        super().__init__(conf_file, device)
        self.orig_size = torch.as_tensor(self.origin_image_shape, device=self.device)

    def _load_conf(self, conf_file):
        super()._load_conf(conf_file)
        self.trt_path = self.conf.get('trt_path')
        self.origin_image_shape = self.conf.get('origin_image_shape')

    def _load_model(self):
        dynamic_shape_value = {'ori_size' : self.origin_image_shape}
        self.model = trt_infer.TRTInference(trt_engine_path=self.trt_path, is_torch_infer=True, dynamic_shape_value=dynamic_shape_value, device=self.device)

    def infer(self, *inputs):
        image_embeddings, inner_state = inputs
        ort_inputs = {
            "image_embeddings": image_embeddings,
            "inner_state": inner_state,
            "ori_size": self.orig_size
        }
        masks = self.model.torch_inference(ort_inputs)[0]
        return masks

class SamRosMaskDecoder(SamRosBase):
    def __init__(self, conf_file, device=0):
        super().__init__(conf_file, device)
        self.mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=self.device)
        self.has_mask_input = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.orig_size = torch.as_tensor(self.origin_image_shape, dtype=torch.int32, device=self.device)
        self.transf = ResizeLongestSide(1024)

        point_grids = build_point_grid(self.point_per_side, partial_y=self.partial_width, partial_x=self.partial_height)
        points_scale = np.array(self.origin_image_shape)[None, :]
        self.coord_grids = torch.as_tensor((point_grids * points_scale)[:,None,:], dtype=torch.float32, device=self.device)
        self.label_input = torch.ones([self.points_per_batch, 1], dtype=torch.float32, device=self.device)
        self.lidar_param = LidarParam()
        # self.lidar_param = LidarParam2()
        self._allocate_buffers()

    def _load_conf(self, conf_file):
        super()._load_conf(conf_file)
        self.trt_path = self.conf.get('trt_path')
        self.points_per_batch = self.conf.get('points_per_batch')
        self.max_point = self.conf.get('max_point')
        self.origin_image_shape = self.conf.get('origin_image_shape')
        self.use_lidar = self.conf.get('use_lidar')
        self.cluster_mode = self.conf.get('cluster_mode')
        self.project_max_point = self.conf.get('project_max_point')
        self.point_per_side = self.conf.get('point_per_side')
        self.partial_width = self.conf.get('partial_width')
        self.partial_height = self.conf.get('partial_height')
        if self.cluster_mode:
            self.pred_iou_thresh = 0.2
            self.stability_score_thresh = 0.75
        else:
            self.pred_iou_thresh = 0.88
            self.stability_score_thresh = 0


    def _load_model(self):
        dynamic_shape = {}
        dynamic_shape_value = {}
        dynamic_shape['point_coords'] = [self.points_per_batch, self.max_point, 2]
        dynamic_shape['point_labels'] = [self.points_per_batch, self.max_point]
        dynamic_shape_value['orig_im_size'] = self.origin_image_shape
        self.model = trt_infer.TRTInference(trt_engine_path=self.trt_path, dynamic_shape=dynamic_shape, dynamic_shape_value=dynamic_shape_value, is_torch_infer=True, device=self.device)

    def _project_by_cluster(self, points):
        labels = points[:,5].astype(int)
        coords = []
        ori_coords = []
        for i in range(labels.max() + 1):
            cluster = points[labels == i]
            n_cluster_point = cluster.shape[0]
            if n_cluster_point > self.project_max_point:
                shuffle = np.random.randint(0, n_cluster_point, size=self.project_max_point)
                cluster = cluster[shuffle]
            cluster = cluster[:, :3].astype(np.float32)
            tmp_point_cloud = np.hstack((cluster, np.ones([len(cluster), 1])))
            cluster = np.dot(tmp_point_cloud, self.lidar_param.transform.T)
            reTransform = cv2.projectPoints(cluster, self.lidar_param.rMat, self.lidar_param.tVec, self.lidar_param.camera_matrix, self.lidar_param.distortion)
            coord = reTransform[0][:, 0].astype(np.int32)
            filter = np.where((coord[:, 0] < self.origin_image_shape[1]) & (coord[:, 1] < self.origin_image_shape[0]) & (coord[:, 0] >= 0) & (coord[:, 1] >= 0))
            coord = coord[filter]
            ori_coord = cluster[filter]
            if coord.shape[0] == 0:
                continue
            coords.append(coord)
            ori_coords.append(ori_coord)
        return ori_coords, coords

    def _allocate_buffers(self):
        drv.init()
        dev = drv.Device(self.device)
        self.ctx_gpu = dev.make_context()

    def __del__(self):
        self.model.cfx.pop()
        self.ctx_gpu.pop()

    def _infer_with_lidar(self, inputs):
        image_embedding, lidar_points = inputs
        ori_coords, coords = self._project_by_cluster(lidar_points)
        n_classes = len(coords)

        orig_lidar_box = np.zeros([n_classes, 4])
        if self.cluster_mode:
            coords_labels, coords_resample = resample_coords(coords, max_point=self.max_point, n_resample=2)
            n_resampled_class = len(coords_resample)
            coord_arr = np.zeros([n_resampled_class, self.max_point, 2], dtype=np.float32)
            label_arr = np.zeros([n_resampled_class, self.max_point], dtype=np.float32)
            cluster_arr = np.zeros([n_resampled_class], dtype=np.int32)
            lidar_boxes = np.zeros([n_resampled_class, 4])
            for i in range(n_classes):
                coord = coords[i]
                orig_lidar_box[i, 0] = np.min(coord[:, 0], axis=0)
                orig_lidar_box[i, 1] = np.min(coord[:, 1], axis=0)
                orig_lidar_box[i, 2] = np.max(coord[:, 0], axis=0)
                orig_lidar_box[i, 3] = np.max(coord[:, 1], axis=0)

            for i in range(n_resampled_class):
                coord = coords_resample[i]
                coord_num = coord.shape[0]
                coord_arr[i, :coord_num, :] = coord
                label_arr[i, :coord_num] = 1
                label_arr[i, coord_num:] = -1
                cluster_arr[i] = coords_labels[i]
                lidar_boxes[i] = orig_lidar_box[coords_labels[i]]

            coords_input = torch.from_numpy(coord_arr).to(self.device)
            labels_input = torch.from_numpy(label_arr).to(self.device)
            lidar_boxes = torch.from_numpy(lidar_boxes).to(self.device)
        else:
            coord = np.concatenate(coords, axis=0)
            n_coord = coord.shape[0]
            coords_input = torch.from_numpy(coord[:,None,:])
            labels_input = torch.ones([n_coord, 1], dtype=torch.float32, device=self.device)
            cluster_arr = np.zeros([n_coord], dtype=np.int32)
        ort_inputs = {
            "image_embeddings": image_embedding,
            "mask_input": self.mask_input,
            "has_mask_input": self.has_mask_input,
            "orig_im_size": self.orig_size
        }
        res = []
        mask_data = MaskData()
        for (coord_input, label_input, lidar_box, cluster_input) in batch_iterator(self.points_per_batch, coords_input, labels_input, lidar_boxes, cluster_arr):
            coord_input = self.transf.apply_coords(coord_input, self.origin_image_shape)
            ort_inputs["point_coords"] = coord_input
            ort_inputs["point_labels"] = label_input
            #print(coord_input)
            #print(label_input)
            # s1=image_embedding.nonzero().shape[0]
            _, iou_preds, masks = self.model.torch_inference(ort_inputs)
            # if torch.isnan(iou_preds).any():
            #     s2=image_embedding.nonzero().shape[0]
            #     print(s1, s2)
            #     _, iou_preds, masks = self.model.torch_inference(ort_inputs)
            #print(iou_preds)
            if self.cluster_mode:
                sam_box = batched_mask_to_box(masks>0)
                lidar_iou=get_lidar_iou(lidar_box, sam_box)
                batch_data = MaskData(
                   masks=masks.flatten(0, 1),
                   iou_preds=iou_preds.flatten(0, 1),
                   lidar_iou=lidar_iou.flatten(0, 1),
                   points=torch.as_tensor(coord_input.repeat([masks.shape[1],1,1])),
                   cluster_label=torch.as_tensor(cluster_input.repeat([masks.shape[1]]))
                #    iou_token_out=iou_token_out.flatten(0, 1)
                )
            else:
                batch_data = MaskData(
                   masks=masks.flatten(0, 1),
                   iou_preds=iou_preds.flatten(0, 1),
                   points=torch.as_tensor(coord_input.repeat([masks.shape[1],1,1])),
                #    iou_token_out=iou_token_out.flatten(0, 1)
                )

            process_data(batch_data, self.use_lidar, self.cluster_mode)
            mask_data.cat(batch_data)

        if len(mask_data.items()) == 0:
            return orig_lidar_box, ori_coords, coords, res
        if self.cluster_mode:
            iou_threshold = 0.5
        else:
            iou_threshold = 0.77

        keep_by_nms = batched_nms(
            mask_data["boxes"].float(),
            mask_data["iou_preds"],
            torch.zeros_like(mask_data["boxes"][:, 0]),  # categories
            iou_threshold=iou_threshold,
        )
        mask_data.filter(keep_by_nms)
        mask_data["segmentations"] = mask_data["masks"]
        mask_data.to_numpy()
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "cluster_label": [mask_data["cluster_label"][idx]]
                # "iou_token_out": mask_data["iou_token_out"][idx],
            }
            res.append(ann)
        return orig_lidar_box, ori_coords, coords, res

    def _infer_image(self, inputs):
        image_embedding = inputs[0]
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_labels": self.label_input,
            "mask_input": self.mask_input,
            "has_mask_input": self.has_mask_input,
            "orig_im_size": self.orig_size
        }
        mask_data = MaskData()
        res = []
        for (coord_input,) in batch_iterator(self.points_per_batch, self.coord_grids):
            coord_input = self.transf.apply_coords(coord_input, self.origin_image_shape)
            ort_inputs["point_coords"] = coord_input
            iou_token_out, iou_preds, masks = self.model.torch_inference(ort_inputs)
            batch_data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                iou_token_out=iou_token_out.flatten(0, 1),
                points=torch.as_tensor(coord_input.repeat([masks.shape[1],1,1])),
            )
            process_data(batch_data)
            mask_data.cat(batch_data)
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
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "iou_token_out": mask_data["iou_token_out"][idx],
            }
            res.append(ann)
        return res

    def infer(self, *inputs):
        if self.use_lidar:
            res = self._infer_with_lidar(inputs)
        else:
            res = self._infer_image(inputs)
        return res

    def get_buffer(self, buffer, check_buffer):
        ipc_handle = buffer
        ipc_handle = bytearray.fromhex(ipc_handle)
        check_handle = bytearray.fromhex(check_buffer)
        x_ptr = drv.IPCMemoryHandle(ipc_handle)
        c_ptr = drv.IPCMemoryHandle(check_handle)
        shape = [1, 256, 64, 64]
        image_embeddings = torch.zeros(shape, dtype=torch.float32, device=self.device)
        check_tensor = torch.zeros([1], dtype=torch.int32, device=self.device)
        drv.memcpy_dtod(image_embeddings.data_ptr(), x_ptr, trt.volume(image_embeddings.shape)*image_embeddings.element_size())
        drv.memcpy_dtod(check_tensor.data_ptr(), c_ptr, 4)
        self.ctx_gpu.synchronize()
        return image_embeddings, check_tensor.item()

