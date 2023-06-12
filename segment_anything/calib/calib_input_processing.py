from segment_anything import sam_trt
import os
import cv2
import torch

device='cuda:0'
sam_trt = sam_trt.SAMTRT(device, use_trt=False)
sam_trt.load_sam()
sam_vit_model = sam_trt.sam.image_encoder
img_dir = "%s/raw_imgs" % os.path.dirname(__file__)
out_dir = "%s/vit_embs" % os.path.dirname(__file__)
img_names = os.listdir(img_dir)
for i, img_name in enumerate(img_names):
    img_path = "%s/%s" % (img_dir, img_name)
    out_path = "%s/%d.pth" % (out_dir, i)
    ori_image = cv2.imread(img_path)
    image = sam_trt.prepare_image(ori_image)
    out = sam_vit_model(image)
    torch.save(out.cpu(), out_path)