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
    for i in range(10):
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
    
def _get_pcd_pair(imgf_dir, lidarf_dir):
    import os
    cache_file = '%s/pcd_pair.txt' % os.path.dirname(__file__)
    macthing_list = []
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            for line in f:
                f_img, f_lidar = line.strip().split(',')
                macthing_list.append((f_img, f_lidar))
        return macthing_list
    imgf_list = sorted(os.listdir(imgf_dir), key=lambda x: float(x.rstrip('.jpg')))
    lidar_list = sorted(os.listdir(lidarf_dir), key=lambda x: float(x.rstrip('.npy')))
    i, j = 0, 0
    matched_lidar_list = []
    while i < len(imgf_list) and j < len(lidar_list):
        t_img = float(imgf_list[i].rstrip('.jpg'))
        t_lidar = float(lidar_list[j].rstrip('.npy'))
        if j+1 < len(lidar_list):
            t_lidar1 = float(lidar_list[j+1].rstrip('.npy'))
            if abs(t_img-t_lidar) > abs(t_img-t_lidar1):
                j+=1
            else:
                matched_lidar_list.append(lidar_list[j])
                i+=1
        else:
            matched_lidar_list.append(lidar_list[j])
            i+=1
    i = 0
    while i < len(matched_lidar_list) - 1:
        if matched_lidar_list[i] == matched_lidar_list[i+1]:
            i+=1
            continue
        else:
            break
    j = len(matched_lidar_list) - 1
    while j > 0:
        if matched_lidar_list[j] == matched_lidar_list[j-1]:
            j-=1
            continue
        else:
            break
    with open(cache_file, 'w') as f:
        for k in range(i+1, j):
            macthing_list.append((imgf_list[k], matched_lidar_list[k]))
            f.write("%s,%s\n" % (imgf_list[k], matched_lidar_list[k]))
    return macthing_list

def _get_trans_param():
    from scipy.spatial.transform import Rotation as R
    # inparm = [1358.080518, 0.0, 987.462437, 0.0, 1359.770396, 585.756872, 0.0, 0.0, 1.0]
    # outparm = [-1.6169,-0.0105,-1.5492, 37.95,-23.58,-25.20]
    # diffcoeffs = np.array([-0.406858, 0.134080, 0.000104, 0.001794, 0.0])
    # cam_intrinsic = np.reshape(np.array(inparm, dtype=np.float64), (3,3))
    # R_matrix = [-90.8, 1, -88.25]
    # r = R.from_euler("xyz", R_matrix, degrees=True)
    # rot = r.as_matrix()
    # trans = np.array(outparm[3:]) * 0.01
    # transform_1 = np.hstack((rot, trans.reshape(3,-1)))
    # transform_1 = np.vstack((transform_1, np.array([0, 0, 0, 1])))
    # transform_1 = np.linalg.inv(transform_1)
    # rotation_z = np.array([0, 0, 0], dtype="float").reshape(3, 1)
    # translation_z = np.array([0, 0, 0], dtype="float").reshape(1, 3)
    camera_matrix = np.array([1358.080518, 0.0, 987.462437,
                          0.0, 1359.770396, 585.756872,
                          0.0, 0.0, 1.0]).reshape((3, 3))

    r = R.from_euler("xyz", [-90.8, 1, -88.25], degrees=True)
    rMat = r.as_matrix()
    tVec = np.array([[0.3795], [-0.2358], [-0.2520]])
    distortion = np.array([[-0.406858, 0.134080, 0.000104, 0.001794, 0.0]], dtype="float")

    # 构建外参矩阵
    transMat = np.identity(4)
    transMat[0:3, 0:3] = rMat

    transMat[0, 3] = tVec[0, 0]
    transMat[1, 3] = tVec[1, 0]
    transMat[2, 3] = tVec[2, 0]

    # 求逆矩阵，将世界坐标系原点换算到Camera, Camera坐标系：x屏幕右，y屏幕下，z屏幕内
    transMat = np.linalg.inv(transMat)

    # 获取更新后的 旋转和平移
    rMat = transMat[0:3, 0:3]
    tVec[0, 0] = transMat[0, 3]
    tVec[1, 0] = transMat[1, 3]
    tVec[2, 0] = transMat[2, 3]
    return rMat, tVec, camera_matrix, distortion

    
def test_lidar():
    from reflect import get_clusters
    import random
    args = parse_args()
    # sam_trt = SAMTRT(args.device, use_trt=True)
    # sam_trt.load_sam()
    imgf_dir = '/raid/qianhao/hby_data/image_data'
    lidarf_dir = '/raid/qianhao/hby_data/fulidars'
    pcd_pairs = _get_pcd_pair(imgf_dir, lidarf_dir)[:300]
    cv2.namedWindow("test", 0)
    cv2.resizeWindow("test", 600, 360)
    max_point = 100
    rMat, tVec, camera_matrix, distortion = _get_trans_param()
    for imgf, lidarf in pcd_pairs:
        ori_img = cv2.imread('%s/%s' % (imgf_dir, imgf))
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        oriHeight, oriWidth, _ = ori_img.shape
        new_img = ori_img
        point_cloud = np.load('%s/%s' % (lidarf_dir, lidarf))
        point_cloud = point_cloud[:, :3]
        point_cam = point_cloud[np.where(point_cloud[:, 0] > 0)[0]]
        # with open('%s/%s' % (lidarf_dir, lidarf)) as f:      
        #     point_list = json.load(f)
        # n_point = len(point_list)
        # point = np.zeros([n_point, 3], dtype=np.float32)
        # for i in range(n_point):
        #     point[i, 0] = point_list[i]['x']
        #     point[i, 1] = point_list[i]['y']
        #     point[i, 2] = -point_list[i]['z']
        pcd, labels = get_clusters(point_cam)
        print(labels.max())
        mask_prompts = []
        for i in range(labels.max() + 1):
            indices = np.where(labels == i)[0]
            cluster = pcd.select_by_index(indices)
            n_cluster_point = np.asarray(cluster.points).shape[0]
            if n_cluster_point > max_point:
                sampling_ratio = max_point / n_cluster_point
                cluster = cluster.random_down_sample(sampling_ratio).remove_duplicated_points()
            points = np.asarray(cluster.points)
            # points = np.hstack((points, np.ones([len(points), 1])))
            # point_cam = np.dot(transform_1, points.T).T  # transform_1 外参
            point_cam = points[:,:3]
            reTransform = cv2.projectPoints(point_cam, rMat, tVec, camera_matrix, distortion)
            pixels = reTransform[0][:, 0].astype(int)
            filter = np.where((pixels[:, 0] < oriWidth) & (pixels[:, 1] < oriHeight) & (pixels[:, 0] >= 0) & (pixels[:, 1] >= 0))
            pixels = pixels[filter]
            mask_prompts.append(pixels)
            color = [random.randint(0, 255) for _ in range(3)]
            for pixel in pixels:
                new_img = cv2.circle(new_img, pixel, 10, color, 10)
        cv2.imshow("test", new_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
                    
            
        
    #     image = sam_trt.prepare_image(ori_img)
    #     res = sam_trt.infer_grid_coord(image)
    #     new_img=ori_img
    #     for r in res:
    #         bbox = r['bbox']
    #         x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    #         pt1=(x, y)
    #         pt2=(x+w, y+h)
    #         new_img = cv2.rectangle(img=new_img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=2)
    #     cv2.imshow("test", new_img)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    test_lidar()

