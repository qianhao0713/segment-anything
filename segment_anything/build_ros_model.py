import os
from segment_anything.sam_ros import SamRosVit, SamRosSeghead, SamRosMaskDecoder

def build_ros_model(node, device, **kwargs):
    conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf')
    if node == 'vit':
        return SamRosVit(
            conf_file='%s/ros_vit.json' % conf_dir,
            device=device
        )
    if node == 'seghead':
        return SamRosSeghead(
            conf_file='%s/ros_seghead.json' % conf_dir,
            device=device,
            **kwargs
        )
    if node == 'mask_decoder':
        return SamRosMaskDecoder(
            conf_file='%s/ros_maskdecoder.json' % conf_dir,
            device=device,
            **kwargs
        )
    raise Exception("invalid node: %s, valid nodes are (vit, seghead, mask_decoder)" % node)