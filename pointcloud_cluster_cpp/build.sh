rm -rf build/
rm -rf lib/
mkdir build
mkdir lib
cd build
cmake .. \
    -DEigen3_INCLUDE_DIR='/usr/include/eigen3'
make
cd ..
mv build/*.so lib/
rm -rf build
