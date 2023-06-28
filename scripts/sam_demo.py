import argparse
import numpy as np
import cv2
from segment_anything.sam_trt import SAMTRT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    return args

def test_single():
    args = parse_args()
    point = np.array([[500, 375]])
    label = np.array([1])

    print("use sam tensort-rt")
    sam_trt = SAMTRT(args.device, use_trt=True)
    sam_trt.generate_masks = False
    sam_trt.load_sam()
    ori_image = cv2.imread('notebooks/images/truck.jpg')
    image = sam_trt.prepare_image(ori_image)

    iou_preds, masks = sam_trt.infer_single_coord(image, point, label)
    sam_trt.show_result(masks, ori_image, point, label, 'truck_single_coord.png')

    print("use sam base")
    sam_trt = SAMTRT(args.device, use_trt=False)
    sam_trt.generate_masks = False
    sam_trt.load_sam()
    ori_image = cv2.imread('notebooks/images/truck.jpg')
    image = sam_trt.prepare_image(ori_image)

    iou_preds, masks = sam_trt.infer_single_coord(image, point, label)

def test_grids():
    args = parse_args()
    
    print("use sam tensort-rt")
    sam_trt = SAMTRT(args.device, use_trt=True)
    sam_trt.load_sam()
    ori_image = cv2.imread('notebooks/images/frame1.jpg')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = sam_trt.prepare_image(ori_image)
    res = sam_trt.infer_grid_coord(image)
    sam_trt.show_result(image=ori_image, anns=res, out_path='frame1_masks.png')

    # print("use sam base")
    # sam_trt = SAMTRT(args.device, use_trt=False)
    # sam_trt.load_sam()
    # ori_image = cv2.imread('notebooks/images/truck.jpg')
    # print(ori_image.shape)
    # image = sam_trt.prepare_image(ori_image)
    # res=sam_trt.infer_grid_coord(image)
    # sam_trt.show_result(image=ori_image, anns=res)

if __name__ == '__main__':
    test_grids()

