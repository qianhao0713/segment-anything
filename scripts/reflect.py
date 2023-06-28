from glob import glob
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import os
import open3d as o3d
import open3d.core as o3c
import time


def get_pointcloud_in_range(points, start, end):


    start_angle = np.deg2rad(start)
    end_angle = np.deg2rad(end)

    angles = np.arctan2(points[:,1], points[:,0])
    angles[angles < 0] += 2*np.pi # 调整角度为0~2*pi之间

    mask = (angles >= start_angle) & (angles <= end_angle)
    selected_points = points[mask]

    return selected_points


def get_distance(points):

    xx = points[:, 0]
    yy = points[:, 1]
    zz = points[:, 2]
    distance = np.sqrt(xx * xx + yy * yy + zz * zz)

    return distance


def get_clusters(points):
    # p1:0-50; p2:310-360
    t=time.time()
    ags = [0, 50, 310, 360]
    p1 = get_pointcloud_in_range(points, ags[0], ags[1])
    p2 = get_pointcloud_in_range(points, ags[2], ags[3])

    newpoints = np.vstack([p1,p2])
    # print(newpoints.shape
    distance = get_distance(newpoints)
    newpoints = newpoints[distance<80]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(newpoints)
    myeps = 0.5
    labels = np.array(pcd.cluster_dbscan(eps=myeps, min_points=10, print_progress=False))
    return pcd, labels

def get_sn_map(image, point, calib, camera_intrinsic, distortion):
    """输入：image:图像数据；
            point：点云数据；
            calib:外参数据；
            camera_intrinsic：内参数据
    """

    savedir = 'results/'

    h, w, _ = image.shape

    ### 根据外参，将点云数据从世界坐标系转换至camera坐标系
    point = np.hstack((point, np.ones([len(point), 1])))
    print('mypoint:', point.shape)
    point_cam = np.dot(calib, point.T).T
    print('point cam 0:', type(point_cam[0,0]))

    ### 过滤掉深度值为负的点，即雷达后方的点
    point_cam = point_cam[:, :3]
    a = np.where(point_cam[:, 2] >= 0)
    point_cam = point_cam[a]

    ### 将camera坐标系的点投影到图像坐标系
    rotation_z = np.array([0, 0, 0], dtype="float").reshape(3, 1)
    translation_z = np.array([0, 0, 0], dtype="float").reshape(1, 3)
    # distortion = np.array([-0.395268, 0.126694, -0.000493, 0.002272], dtype="float")
    reTransform = cv2.projectPoints(point_cam, rotation_z, translation_z, camera_intrinsic, distortion)
    pixel = reTransform[0][:, 0].astype(int)

    ### 过滤掉超出图像大小范围的点
    filter = np.where((pixel[:, 0] < w) & (pixel[:, 1] < h) & (pixel[:, 0] >= 0) & (pixel[:, 1] >= 0))
    pixel = pixel[filter]

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    for i in range(len(pixel)):
        image = cv2.circle(image, (pixel[i, 0], pixel[i ,1]), 1, (0, 0, 255), 1)
    # cv2.imwrite('pointcloud_sim.png', image)
    picname = savedir + 'img_' + imgname.replace(imgdir, '').replace('.jpg', '') + '_lidar_' + pointname.replace(pointdir, '').replace('.npy', '') + '.png'
    # print(picname)
    # print(image.shape)
    cv2.imwrite(picname, image)

if __name__ == '__main__':
    inparm = [1358.080518, 0.0, 987.462437, 0.0, 1359.770396, 585.756872, 0.0, 0.0, 1.0]
    outparm = [-1.6169,-0.0105,-1.5492, 37.95,-23.58,-25.20]
    diffcoeffs = np.array([-0.406858, 0.134080, 0.000104, 0.001794, 0.0])
    
    cam_intrinsic = np.reshape(np.array(inparm, dtype=np.float64), (3,3))
    
    R_matrix = [round(math.degrees(a)) for a in outparm[:3]]
    print('原始旋转矩阵：', R_matrix)
    ### 计算并且得到外参
    
    # R_matrix = [-90, 0, 90]
    
    R_matrix = [-90.8, 1, -88.25]
    print('当前旋转矩阵：', R_matrix)
    r = R.from_euler("xyz", R_matrix, degrees=True)
    rot = r.as_matrix()
    
    trans = np.array(outparm[3:]) * 0.01
    transform_1 = np.hstack((rot, trans.reshape(3,-1)))
    transform_1 = np.vstack((transform_1, np.array([0, 0, 0, 1])))
    transform_1 = np.linalg.inv(transform_1)
    
    pointdir = 'fulidars/'
    imgdir = 'fuimgs3/'
    
    imgname = imgdir +  '1678244635.8501132' + '.jpg'
    # pointname = pointls[num2]
    pointname = pointdir + '1678244635.8547657' + '.npy'
    
    print(imgname)
    print(pointname)
    
    img = cv2.imread(imgname)
    # point = np.load(pointname)
    point = np.load(pointname)[:,:3]
    # print('point size:', point.shape)
    # img = cv2.undistort(img, cam_intrinsic, diffcoeffs)
    print('point 0',type(point[0,0]))
    
    t1 = time.time()
    
    pcd, labels = get_clusters(point)
    
    # print(pcd)
    
    # 跟踪结果：中心点-速度、尺寸-长宽高、速度-中心点差值
    # ctptls = []
    # sizels = []
    # bboxls = []
    # newboxls = []
    
    aabbls = []
    ctptls = []
    wlhls = []
    
    print('类别:', max(labels))
    
    for i in range(max(labels)+1):
        indices = np.where(labels == i)[0]
        cluster = pcd.select_by_index(indices)
    
        aabb = cluster.get_axis_aligned_bounding_box()
    
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
    
        one = [min_bound[0], min_bound[1], min_bound[2], max_bound[0], max_bound[1], max_bound[2]]
    
        w, l, h = one[3]-one[0], one[4]-one[1], one[5]-one[2]
    
        if w*l*h < 1:
            continue
        
        aabbls.append(one)
    
        wlhls.append([w, l, h])
    
        xc = (one[0]+one[3])/2
        yc = (one[1]+one[4])/2
        zc = (one[2]+one[5])/2
    
        ctptls.append([xc,yc,zc])
    
    
    # 中心点合集列表 转 numpy
    ctpts = np.array(ctptls)
    print(type(ctpts[0,0]))
    
    # 世界坐标系转camera坐标系
    point1 = np.hstack((ctpts, np.ones([len(ctpts), 1])))
    
    
    # # 相机视角点云所有点 转 numpy
    # pcd_points = np.asarray(pcd.points)
    # # 世界坐标系转camera坐标系
    # point1 = np.hstack((pcd_points, np.ones([len(pcd_points), 1])))
    
    
    print('mypoint:', point1.shape)
    point_cam = np.dot(transform_1, point1.T).T  # transform_1 外参
    
    point_cam = point_cam[:,:3]
    
    rotation_z = np.array([0, 0, 0], dtype="float").reshape(3, 1)
    translation_z = np.array([0, 0, 0], dtype="float").reshape(1, 3)
    reTransform = cv2.projectPoints(point_cam, rotation_z, translation_z, cam_intrinsic, diffcoeffs)
    pixel = reTransform[0][:, 0].astype(int)
    
    cv2.imshow('img', img)
    print('speed time:', time.time()-t1)
    cv2.waitKey(0)
    
    get_sn_map(image=img, point=point, calib=transform_1, camera_intrinsic=cam_intrinsic, distortion=diffcoeffs)
