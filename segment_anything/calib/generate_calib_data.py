import torch
import os
from segment_anything import sam_trt
import cv2


cur_dir = os.path.abspath(os.path.dirname(__file__))
img_dir = '%s/raw_calib_img' % cur_dir
tensor_dir = '%s/calib_data/vit' % cur_dir
img_paths = os.listdir(img_dir)
device = 'cuda:2'
sam = sam_trt.SAMTRT(device=device, use_trt=False)
sam.load_sam()
batch_size = 1
concat_vit_embs = None
batch_index = 0
embs = []
for i, img_path in enumerate(img_paths):
    ori_image = cv2.imread('%s/%s' % (img_dir, img_path))
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = sam.prepare_image(ori_image)
    vit_emb = sam.infer_vit_embedding({"image": image})
    embs.append(vit_emb.clone())
    batch_index += 1
    if batch_index == batch_size:
        batch_index = 0
        embs_concat = torch.cat(embs, dim=0)
        print(embs_concat.shape)
        torch.save(embs_concat, '%s/%s.pt' % (tensor_dir, i))
        print("progress: %d / %d" % (i, len(img_paths)))
        embs = []