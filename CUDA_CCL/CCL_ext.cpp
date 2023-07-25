#include <torch/extension.h>
#include "CCL.h"
#include "timer.h"
#include "utils.hpp"

void torch_ccl(torch::Tensor& label, const torch::Tensor& mask, size_t n_row, size_t n_col) {
    GpuTimer timer;
	timer.Start();
    cudaSetDevice(mask.device().index());
    auto shape = mask.sizes();
    if (shape.size() == 2) {
        connectedComponentLabeling((signed int*) label.data_ptr(), (unsigned char*) mask.data_ptr(), n_col, n_row);
    }
    if (shape.size() == 3) {
        auto rep = shape[0];
        for (int i = 0; i < rep; i++) {
            connectedComponentLabeling((signed int*) label.index({i}).data_ptr(), (unsigned char*) mask.index({i}).data_ptr(), n_col, n_row);
        }
    }
    if (shape.size() == 4) {
        auto rep0 = shape[0];
        auto rep1 = shape[1];
        for (int i = 0; i < rep0; i++) {
            for (int j = 0; j < rep1; j++) {
                connectedComponentLabeling((signed int*) label.index({i, j}).data_ptr(), (unsigned char*) mask.index({i, j}).data_ptr(), n_col, n_row);
            }  
        }
    }
    // cudaDeviceSynchronize();
    cudaSetDevice(0);
    timer.Stop();
    // std::cout << "GPU code ran in: " << timer.Elapsed() << "ms" << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_ccl", &torch_ccl, "torch_ccl kernel warpper");
}
