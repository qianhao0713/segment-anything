rm -rf build/
rm -rf lib/
mkdir build
mkdir lib
cd build
cmake .. \
    -Dpybind11_DIR='/home/qianhao/third_party/pybind11/build' \
    -DOpenCV_DIR='/home/qianhao/third_party/opencv/build' \
	-DEigen3_INCLUDE_DIR='/usr/include/eigen3'
make
cd ..
mv build/*.so lib/
rm -rf build