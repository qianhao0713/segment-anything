export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qianhao/workspace/TensorRT-8.6.1.6/lib/
cd $(dirname $0)/..
# export model to tensorRT
python scripts/export_tensorrt.py
# ./bin/trtexec --onnx=./weights/sam_image_encoder_fold.onnx --saveEngine=./weights/sam_image_encoder.trt
# ./bin/trtexec --onnx=./weights/sam_vit_l_single_mask_decoder_int8.onnx --saveEngine=./weights/sam_vit_l_single_mask_decoder_int8.trt --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:64x1x2,point_labels:64x1 --maxShapes=point_coords:1024x2x2,point_labels:1024x1 --verbose
