#ifndef _DRIVABLE_AREA_H_
#define _DRIVABLE_AREA_H_


#include "point_types.h"
#include "rclcpp/rclcpp.hpp"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/point_cloud.h>
#include <sensor_msgs/msg/point_cloud2.h>
#include "pcl_conversions/pcl_conversions.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <visualization_msgs/msg/marker.hpp>
//#include "perception_msgs/msg/obpoint.hpp"
//#include "perception_msgs/msg/obstacle_mat.hpp"

class Cluster: public rclcpp::Node
{
private:

  void AllPointCloudCluster(const sensor_msgs::msg::PointCloud::SharedPtr scan, unsigned npoints_);

  double m_per_cell_;
  double height_diff_threshold_;
  bool full_clouds_;
        
  cv::Mat obs_image;
  cv::Mat range_image;
  cv::Mat Mask_image;
  cv::Mat allim;
  cv::Mat LastRoadArea;
  cv::Mat LastReachableArea;
  cv::Mat obs_image_send;
  cv::Mat obs_image_send_temp;
  cv::Mat AllMask;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr sub;
  //rclcpp::Publisher<perception_msgs::msg::ObstacleMat>::SharedPtr obspub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_marker;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_marker2;

  public:
  //IplImage* ObsImage;
  //Obstacle2DNameSpace::pointcloud **mPoint;//点云存储

  Cluster(std::string name):Node(name)
  {
      //m_per_cell_=0.2;
      //full_clouds_=false;
      //height_diff_threshold_=0.25;

      //obs_image =  cv::Mat::zeros(750, 500,CV_8UC1);
      //range_image =  cv::Mat::zeros(128, 1800,CV_8UC1);
      //ObsImage = cvCreateImage(cvSize(500,750),8,1); 
      //LastRoadArea = cv::Mat::zeros(750, 500,CV_8UC1);
      //LastReachableArea = cv::Mat::zeros(750, 500,CV_8UC1);
      //obs_image_send = cv::Mat::zeros(1200, 800,CV_8UC1);
      //obs_image_send_temp = cv::Mat::zeros(1200, 800,CV_8UC1);
      

      using std::placeholders::_1;
      sub = this->create_subscription<sensor_msgs::msg::PointCloud>("pointcloud_ground_segmentation", 1, std::bind(&Cluster::processData, this, std::placeholders::_1));
      publisher_marker = this->create_publisher<visualization_msgs::msg::Marker>("point",rclcpp::QoS(1));
      publisher_marker2 = this->create_publisher<visualization_msgs::msg::Marker>("box",rclcpp::QoS(1));
      //obspub = this->create_publisher<perception_msgs::msg::ObstacleMat>("ObstacleMat", 1);
      
      //mPoint=new Obstacle2DNameSpace::pointcloud *[128];
      //for (size_t i = 0; i < 128; i++)
      //{
      //  mPoint[i]=new Obstacle2DNameSpace::pointcloud[1800];
      //}

  };
  ~Cluster();
   //==============================点云回调函数===========================
  void processData(const sensor_msgs::msg::PointCloud::SharedPtr scan);
};

#endif
