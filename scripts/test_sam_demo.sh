STARLIGHT_ROOT_PATH=/home/qianhao/StarLight
LIB_TENSORRT_PATH=$STARLIGHT_ROOT_PATH/TensorRT-8.6.1.6/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_TENSORRT_PATH
cd $(dirname $0)/..
device=0
if [ $# -ge 1 ];
then
	device=$1
fi
export CUDA_DEVICE=$device
python scripts/sam_demo.py --device $device
