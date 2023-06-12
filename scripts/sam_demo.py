import argparse
import numpy as np
import cv2
from segment_anything.sam_trt import SAMTRT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_coord_input', action='store_true')
    parser.add_argument('--exclude_postprocess', action='store_true')
    # parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--encoder_trt', type=str, default='weights/sam_image_encoder.trt', help='sam image encoder trt path')
    parser.add_argument('--decoder_trt', type=str, default='weights/sam_single_mask_mask_decoder_fold.trt', help='sam mask decoder trt path')
    # parser.add_argument('--decoder_trt', type=str, default='weights/sam_mask_decoder_fold.trt', help='sam mask decoder trt path')
    parser.add_argument('--points_per_side', type=int, default=32, help='point input num in one-side of a picture for mask-generation, total num is n_points * n_points')
    parser.add_argument('--points_per_batch', type=int, default=64, help='point input in one batch for mask-generation')
    parser.add_argument('--box_nms_thresh', type=float, default=0.77)
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

