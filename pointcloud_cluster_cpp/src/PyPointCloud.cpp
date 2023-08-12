#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <chrono>
#include <boost/python.hpp>
#include <omp.h>
#include <sensor_msgs/msg/point_cloud.h>
#include <sensor_msgs/msg/point_cloud2.h>
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <rclcpp/serialization.hpp>
#include "PointCloudTool.hpp"
#include "tool.hpp"

namespace py = pybind11;

class PyPointCloud {
public:
    PyPointCloud() {
        mcloud = new pointCloud;
    }
    Eigen::MatrixXd execute_cluster(Eigen::MatrixXd& mat) {
        init();
        load_input(mat);
        newtool.NewGroundSeg(mcloud);
        std::vector< std::vector< std::vector<pointX> > > AllClusters = newtool.PointCloudCluster(mcloud);
        vector<OneCluster> FinalCLuster= newtool.CombineClusterResult(&AllClusters);
        assign_label(mat, AllClusters, FinalCLuster);
        return mat;
    }

private:
    pointCloud *mcloud;
    PointcloudTool newtool;

    void init() {
        for (int l = 0; l < LINE; l++) {
            for (int c = 0; c < CIRCLEMAXLEN; c++) {
                mcloud->mptclout[l][c].x = 0;
                mcloud->mptclout[l][c].y = 0;
                mcloud->mptclout[l][c].z = 0;
                mcloud->mptclout[l][c].d = 0;
                mcloud->mptclout[l][c].isused = 0;  
                mcloud->mptclout[l][c].type = 0;
            }
        }
        mcloud->circlelen = 1800;
    }

    void load_input(Eigen::MatrixXd& mat) {
        #pragma omp parallel for
        for (auto i = 0; i < mat.rows(); i++) {
            float x = mat(i, 0);
            float y = mat(i, 1);
            float z = mat(i, 2);
            int circlept = (int) mat(i, 3);
            int line = (int) mat(i, 4);
            mcloud->mptclout[line][circlept].x = x;
            mcloud->mptclout[line][circlept].y = y;
            mcloud->mptclout[line][circlept].z = z;
            mcloud->mptclout[line][circlept].isused=1;
            mcloud->mptclout[line][circlept].ori_id = i;
            mcloud->mptclout[line][circlept].type = 20; 
            mat(i, 5) = -1.0;
        }
    }


    void assign_label(Eigen::MatrixXd& mat, std::vector< std::vector< std::vector<pointX>>>& AllClusters, vector<OneCluster>& FinalCLuster) {
        for (int i = 0; i < FinalCLuster.size(); i++) {
            auto& one_cluster = FinalCLuster[i];
            auto& point_indexes = one_cluster.PointIndex;
            for (auto& point_index: point_indexes) {
                int r_index = point_index.x;
                int c_index = point_index.y;
                auto& sub_cluster = AllClusters[r_index][c_index];
                for (auto& point: sub_cluster) {
                    int ori_id = point.ori_id;
                    mat(ori_id, 5) = (double)i;
                }
            }
        }
    }
};

PYBIND11_MODULE(pointcloud_cluster, m) {
    py::class_<PyPointCloud>(m, "PyPointCloud")
        .def(py::init<>())
        .def("execute_cluster", &PyPointCloud::execute_cluster);
}