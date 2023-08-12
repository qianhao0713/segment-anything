from segment_anything.build_ros_model import build_ros_model
import argparse
import utils
import cv2
import sys
import time

# IMG_DIR = '/home/user/gitcode/depends/segment-anything/data/hby_data/image_data'
# LIDAR_DIR = '/home/user/gitcode/depends/segment-anything/data/hby_data/fulidars'
IMG_DIR = '/home/gywrc-s1/qianhao/data/hby_data_backup/image_data'
LIDAR_DIR = '/home/gywrc-s1/qianhao/data/hby_data_backup/fulidars'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args

def show_image(image):
    cv2.namedWindow("test", 0)
    cv2.resizeWindow("test", 1000, 600)
    cv2.imshow("test", image)
    waitKey = cv2.waitKey(25)
    if waitKey & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)
    if waitKey & 0xFF == ord('p'):
        while True:
            if cv2.waitKey(25) & 0xFF == ord('c'):
                return

def test_freespace():
    import torch
    import numpy as np

    args = parse_args()
    device = args.device
    pcd_pairs = utils.get_pcd_pair(IMG_DIR, LIDAR_DIR)
    mask_color=[255, 0, 0]

    # build tensorrt model
    vit_model = build_ros_model('vit', device)
    seghead_model = build_ros_model('seghead', device)

    #infer
    for imgf, _ in pcd_pairs:
        ori_image = cv2.imread('%s/%s' % (IMG_DIR, imgf))
        bgr_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        image_embedding, inner_state = vit_model.infer(bgr_image)
        masks = seghead_model.infer(image_embedding, inner_state)
        #show result
        pred_img = torch.softmax(masks, dim=1).cpu().float().detach().numpy()[0][1]
        index = np.where(pred_img > 0.95)
        ori_image[index] = mask_color
        # show_image(ori_image)

def test_lidar():
    import numpy as np
    from pointcloud_cluster_cpp.lib import pointcloud_cluster
    args = parse_args()
    device = args.device
    pcd_pairs = utils.get_pcd_pair(IMG_DIR, LIDAR_DIR)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter('output.avi', fourcc, 10.0, (1920,  1080))
    vit_model = build_ros_model('vit', device)
    mask_decoder_model = build_ros_model('mask_decoder', device)
    pointcloud_cluster_tool = pointcloud_cluster.PyPointCloud()
    for imgf, lidarf in pcd_pairs:
        ori_image = cv2.imread('%s/%s' % (IMG_DIR, imgf))
        point_cloud = np.load('%s/%s' % (LIDAR_DIR, lidarf))
        bgr_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        points = utils.prepare_lidar(point_cloud)
        points = pointcloud_cluster_tool.execute_cluster(points)
        points = points[points[:,5]>=0]
        image_embedding, _ = vit_model.infer(bgr_image)
        coords, res = mask_decoder_model.infer(image_embedding, points)
        utils.show_lidar_result(ori_image, coords=coords, res=res, show_mask=True, video_writer=vw)

def test_multiprocess():
    import multiprocessing as mp
    import numpy as np
    from multiprocessing import Process
    from multiprocessing import Queue
    def vit_process(q):
        import json
        args = parse_args()
        device = args.device
        pcd_pairs = utils.get_pcd_pair(IMG_DIR, LIDAR_DIR)
        vit_model = build_ros_model('vit', device)
        for imgf, lidarf in pcd_pairs[:10]:
            ori_image = cv2.imread('%s/%s' % (IMG_DIR, imgf))
            bgr_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
            image_embedding, _ = vit_model.infer(bgr_image)
            vit_model.store_buffer(image_embedding)
            vit_res_info = vit_model.result_info()
            vit_res_info["imgf"] = imgf
            vit_res_info["lidarf"] = lidarf
            msg = json.dumps(vit_res_info)
            print(msg)
            q.put(msg)

    def decoder_process(q):
        import json
        from pointcloud_cluster_cpp.lib import pointcloud_cluster
        args = parse_args()
        device = args.device
        mask_decoder_model = build_ros_model('mask_decoder', device)
        pointcloud_cluster_tool = pointcloud_cluster.PyPointCloud()
        msg = q.get(True)
        res_info = json.loads(msg)
        image_embeddings = mask_decoder_model.get_buffer(res_info)
        lidarf = res_info['lidarf']
        point_cloud = np.load('%s/%s' % (LIDAR_DIR, lidarf))
        points = utils.prepare_lidar(point_cloud)
        points = pointcloud_cluster_tool.execute_cluster(points)
        points = points[points[:,5]>=0]
        coords, res = mask_decoder_model.infer(image_embeddings, points)

    q=Queue()
    mp.get_context('spawn')
    p1=Process(target=vit_process, args=(q,))
    p2=Process(target=decoder_process, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()


if __name__ == '__main__':
    test_lidar()
