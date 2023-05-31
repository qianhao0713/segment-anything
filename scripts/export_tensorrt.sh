STARLIGHT_ROOT_PATH=/home/qianhao/StarLight
LIB_TENSORRT_PATH=$STARLIGHT_ROOT_PATH/TensorRT-8.6.1.6/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_TENSORRT_PATH
# export EXPORT_MODULE_PATH=$STARLIGHT_ROOT_PATH/algorithms/compression/lib

cd $(dirname $0)/..
# export model to tensorRT
python scripts/export_tensorrt.py
# ./bin/trtexec --onnx=./weights/sam_image_encoder_fold.onnx --saveEngine=./weights/sam_image_encoder.trt
#./bin/trtexec --onnx=./weights/sam_mask_decoder_fold.onnx --saveEngine=./weights/sam_mask_decoder_2.trt --minShapes=point_coords:1x1x2,point_labels:1x1,orig_im_size:1x1 --optShapes=point_coords:1x2x2,point_labels:1x2,orig_im_size:1200x1800 --maxShapes=point_coords:1x1024x2,point_labels:1x1024,orig_im_size:2400x3600 
