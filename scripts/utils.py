import os
import random
random_colors = [[random.randint(0,255) for _ in range(3)] for _ in range(100)]

def get_pcd_pair(imgf_dir, lidarf_dir):
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

def show_lidar_result(img, coords, res, show_mask=False, video_writer=None):
    import cv2, random, sys
    for coord in coords:
        color = [255,0,0]
        for pixel in coord:
            pixel = pixel.astype('int32')
            img = cv2.circle(img, pixel, 3, color, 3)
    for i in range(len(res)):
        color = random_colors[i]
        bbox=[int(x) for x in res[i]['bbox']]
        x, y, w, h = bbox
        img = cv2.rectangle(img, [x, y], [x+w, y+h], color, 2)
        if show_mask:
            mask=res[i]['segmentation']
            img[mask]=color
    cv2.imshow("test", img)
    if video_writer:
        video_writer.write(img)   
    waitKey = cv2.waitKey(25)
    if waitKey & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        video_writer.release()
        sys.exit(0)
    if waitKey & 0xFF == ord('p'):
        while True:
            if cv2.waitKey(25) & 0xFF == ord('c'):
                break