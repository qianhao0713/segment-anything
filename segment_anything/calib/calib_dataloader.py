from onnxruntime.quantization.calibrate import CalibrationDataReader
from segment_anything.utils.amg import build_point_grid
import os
import torch
import numpy as np

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
        self.coord_grids = (point_grids * points_scale)[:,None,:]
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
        
if __name__ == '__main__':
    dataloader = SamCalibrationDataReader()
    emb = None
    for i, data in enumerate(dataloader):
        print(data["point_coords"])
