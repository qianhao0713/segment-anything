from onnxruntime.quantization.calibrate import CalibrationDataReader
from segment_anything.utils.amg import build_point_grid
import os
import torch
import numpy as np
import tensorrt as trt

class SamCalibrationDataReader(CalibrationDataReader):
    def __init__(self) -> None:
        super().__init__()
        vit_emb_dir = "%s/vit_embs" % os.path.dirname(__file__)
        vit_emb_filenames = os.listdir(vit_emb_dir)
        self.vit_emb_path = ["%s/%s" % (vit_emb_dir, vit_emb_filename) for vit_emb_filename in vit_emb_filenames]
        self.vit_index = 0
        self.point_index = 0

        self.batch_size = 64
        self.point_per_side = 32
        self.max_point_index = self.point_per_side ** 2 // self.batch_size
        self.max_file = 10
        point_grids = build_point_grid(self.point_per_side)
        origin_image_shape = [1080, 1920]
        points_scale = np.array(origin_image_shape)[None, :]
        self.coord_grids = (point_grids * points_scale)[:,None,:].astype(np.float32)
        label_input = np.ones([self.batch_size, 1], dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros(1, dtype=np.float32)
        orig_im_size = np.array(origin_image_shape, dtype=np.int32)
        vit_emb = torch.load(self.vit_emb_path[self.vit_index], map_location='cpu').detach().numpy()
        self.ort_input = {
            "image_embeddings": vit_emb,
            "point_labels": label_input,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size
        }

    def get_next(self) -> dict:
        print("calibrate input: %d/%d" % (self.point_index + self.max_point_index * self.vit_index, self.max_file * self.max_point_index))
        if self.point_index >= 16:
            self.point_index = 0
            self.vit_index += 1
            if self.vit_index >= self.max_file:
                return None
            else:
                vit_emb = torch.load(self.vit_emb_path[self.vit_index], map_location='cpu').detach().numpy()
                self.ort_input["image_embeddings"] = vit_emb
        coord_input = self.coord_grids[self.point_index * self.batch_size : (self.point_index + 1) * self.batch_size]
        self.ort_input["point_coords"] = coord_input
        self.point_index += 1
        return self.ort_input

class SAMCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, device):
        trt.IInt8EntropyCalibrator2.__init__(self)
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        self.dataloader = SamCalibrationDataReader()
        self.batch_size = 1
        self.cache_file = "%s/int8calib.cache" % cur_dir
        self.device = device

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        result = []
        ort_inputs = self.dataloader.get_next()
        if ort_inputs == None:
            return None
        for name in names:
            result.append(torch.as_tensor(ort_inputs[name], device=self.device).data_ptr())
        return result

    def read_calibration_cache(self):
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        
if __name__ == '__main__':
    dataloader = SamCalibrationDataReader()
    emb = None
    for i, data in enumerate(dataloader):
        print(data["point_coords"])
