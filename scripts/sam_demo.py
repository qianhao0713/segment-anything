import argparse
import numpy as np
import cv2
from segment_anything.sam_trt import SAMTRT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--show_img', type=bool, default=True)
    args = parser.parse_args()
    return args

def test_single(args):
    point = np.array([[1000, 800]])
    label = np.array([1])

    print("use sam tensort-rt")
    sam_trt = SAMTRT(args.device, use_trt=True)
    sam_trt.generate_masks = False
    sam_trt.load_sam()
    ori_image = cv2.imread('notebooks/images/frame4.jpg')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = sam_trt.prepare_image(ori_image)
    iou_preds, masks = sam_trt.infer_single_coord(image, point, label)
    print(masks.shape)
    sam_trt.show_result(masks, ori_image, point, label, out_path='frame4_single.png')

    # print("use sam base")
    # sam_trt = SAMTRT(device=args.device, use_trt=False)
    # sam_trt.generate_masks = False
    # sam_trt.load_sam()
    # ori_image = cv2.imread('notebooks/images/frame1.jpg')
    # ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    # image = sam_trt.prepare_image(ori_image)
    # iou_preds, masks = sam_trt.infer_single_coord(image, point, label)
    # # sam_trt.show_result(masks, ori_image, point, label, out_path='frame1_single.png')

def test_grids(args):
    print("use sam tensort-rt")
    sam_trt = SAMTRT(args.device, use_trt=True)
    sam_trt.load_sam()
    ori_image = cv2.imread('notebooks/images/frame4.jpg')
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = sam_trt.prepare_image(ori_image)
    for i in range(10):
        res = sam_trt.infer_grid_coord(image)
    sam_trt.show_result(image=ori_image, anns=res, out_path='frame4_masks.png')

    # print("use sam base")
    # sam_trt = SAMTRT(device=args.device, use_trt=False)
    # sam_trt.load_sam()
    # ori_image = cv2.imread('notebooks/images/frame1.jpg')
    # image = sam_trt.prepare_image(ori_image)
    # res=sam_trt.infer_grid_coord(image)
    # sam_trt.show_result(image=ori_image, anns=res)

    
def test_lidar():
    import utils
    args = parse_args()
    sam_trt = SAMTRT(args.device, use_trt=True, use_lidar=True)
    sam_trt.load_sam()
    imgf_dir = '/raid/qianhao/hby_data_0308/image_data'
    lidarf_dir = '/raid/qianhao/hby_data_0308/fulidars'
    pcd_pairs = utils.get_pcd_pair(imgf_dir, lidarf_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter('output.avi', fourcc, 10.0, (1920,  1080))
    if args.show_img:
        cv2.namedWindow("test", 0)
        cv2.resizeWindow("test", 1000, 600)
    for imgf, lidarf in pcd_pairs:
        ori_img = cv2.imread('%s/%s' % (imgf_dir, imgf))
        point_cloud = np.load('%s/%s' % (lidarf_dir, lidarf))
        image = sam_trt.prepare_image(ori_img)
        coords, res = sam_trt.infer_lidar_points(image, point_cloud)
        if args.show_img:
            utils.show_lidar_result(ori_img, coords, res, show_mask=True, video_writer=vw)
    cv2.destroyAllWindows()
    vw.release()

if __name__ == '__main__':
    test_lidar()

