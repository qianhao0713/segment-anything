# STARLIGHT_ROOT_PATH=/home/qianhao/StarLight
# LIB_TENSORRT_PATH=$STARLIGHT_ROOT_PATH/TensorRT-8.6.1.6/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_TENSORRT_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/raid/qianhao/TensorRT-8.6.1.6/lib/
cd $(dirname $0)/..
device=0
mode='grids'
if [ $# -ge 1 ];
then
	device=$1
fi
if [ $# -ge 2 ];
then
	mode=$2
fi
export CUDA_DEVICE=$device
python scripts/sam_demo.py --device $device --trt_mode $mode
