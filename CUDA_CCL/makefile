##############################################################
NVCC=nvcc
SRC = ./main.cu
CUDAFLAGS= -arch=sm_80
RM=/bin/rm -f

LDFLAGS += -L/home/qianhao/third_party/bin/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -I/home/qianhao/third_party/bin/include/opencv4/
IFLAGES += -I/home/qianhao/third_party/bin/include/opencv4/

main: main.o  CCL.o
	${NVCC} ${CUDAFLAGS} $(LDFLAGS) -o main main.o CCL.o

main.o: CCL.cuh utils.hpp main.cu
	$(NVCC) $(CUDAFLAGS) ${IFLAGES} -std=c++11 -c main.cu

CCL.o: CCL.cuh CCL.cu reduction.cuh
	${NVCC} ${CUDAFLAGS} -std=c++11 -c CCL.cu

clean:
	${RM} *.o main
