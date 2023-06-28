/** 
        obs_image:(左上坐标系)   
      
        --------------------------------->  x
       |

       |                       ^  
                               | x 
       |                       |
                               |
       |              y <------    (point cloud: 单位：米 )  

       |
        y

/*=================================
type:  0  :无效
       10 :地面
       20 :普通障碍物
==================================*/

#include "cluster.h"
#include "PointCloudTool.hpp"
#include "tool.hpp"



#define _MIN(x,y) ((x) < (y) ? (x) : (y))
#define _MAX(x,y) ((x) > (y) ? (x) : (y))


 //点云对象
pointCloud *mcloud;
bool IsDebug=0;
bool IsShow=0;




void Cluster::processData(const sensor_msgs::msg::PointCloud::SharedPtr scan)
{
  cout<<"0"<<endl;
  double now_sec = rclcpp::Clock().now().seconds();
  size_t npoints = scan->points.size();
  size_t obs_count=0;
  size_t empty_count=0;

  AllPointCloudCluster(scan,npoints);

  /*
  perception_msgs::msg::ObstacleMat temp;
  perception_msgs::msg::Obpoint  t;
  for (size_t i = 0; i < 1200; i++)
      for (size_t j = 0; j < 800; j++)
     if(result.ptr<uchar>(i)[j]==255)
     {
       t.x=i;
       t.y=j;
       temp.points.push_back(t);
     }
  obspub->publish(temp);
  */

  //printf("size_obs: %d\n",temp.points.size());
  printf("time: %f ：  Cluster is ok \n",now_sec);

}


void Cluster::AllPointCloudCluster(const sensor_msgs::msg::PointCloud::SharedPtr scan,unsigned npoints_)
{

    for (int l = 0; l < LINE; l++)
     for (int c = 0; c < CIRCLEMAXLEN; c++)
      {
        mcloud->mptclout[l][c].x = 0;
        mcloud->mptclout[l][c].y = 0;
        mcloud->mptclout[l][c].z = 0;
        mcloud->mptclout[l][c].d = 0;
        mcloud->mptclout[l][c].isused = 0;  
        mcloud->mptclout[l][c].type = 0;
      }
    cout<<"1"<<endl;
    int getsize = npoints_-1;
    int getcirclelen = 1800;//getsize / LINE; //获得点云一圈点数

    mcloud->circlelen = getcirclelen; //获取点云一圈的真正长度

    int line; int circlept;
    for (size_t i = 0; i < getsize; i++)
    {
        
        circlept=scan->channels[3].values[i]; 
        line=scan->channels[2].values[i]; 
        
        mcloud->mptclout[line][circlept].x = scan->points[i].x;
        mcloud->mptclout[line][circlept].y = scan->points[i].y;
        mcloud->mptclout[line][circlept].z = scan->points[i].z;
        mcloud->mptclout[line][circlept].d = scan->channels[0].values[i];  //距离
        mcloud->mptclout[line][circlept].type = scan->channels[1].values[i]; 
        //mcloud->mptclout[line][circlept].c = circlept; 
        //mcloud->mptclout[line][circlept].l = line; 
        mcloud->mptclout[line][circlept].gridx = scan->channels[4].values[i]; 
        mcloud->mptclout[line][circlept].gridy = scan->channels[5].values[i]; 
        mcloud->mptclout[line][circlept].lowest = scan->channels[6].values[i]; 
        mcloud->mptclout[line][circlept].isused=1;

        //将无效的点云数据通通赋值为0
        if (mcloud->mptclout[line][circlept].x == 0 &&
            mcloud->mptclout[line][circlept].y == 0 &&
            mcloud->mptclout[line][circlept].z == 0){
            mcloud->mptclout[line][circlept].d = 0;
            mcloud->mptclout[line][circlept].type = 0;
        }
    }
    cout<<"2"<<endl;
        visualization_msgs::msg::Marker points, box;
    
        PointcloudTool newtool;
        //newtool.MaskForCluster=MaskForCluster.clone();
        //newtool.NewGroundSeg(mcloud); //地面滤除
    cout<<"21"<<endl;
    newtool.NewGroundSeg(mcloud);
    cout<<"22"<<endl;
        std::vector< std::vector< std::vector<pointX> > > AllClusters = newtool.PointCloudCluster(mcloud);
    cout<<"3"<<endl;
        vector<OneCluster> FinalCLuster= newtool.CombineClusterResult(&AllClusters);
        newtool.ShowResultWithHull(points,box);

        publisher_marker->publish(points);
        publisher_marker2->publish(box);

}


Cluster::~Cluster() {
  //cvReleaseImage(&ObsImage);
}


int main(int argc, char **argv)
{
 
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Cluster>("cluster");
    mcloud = new pointCloud;
    rclcpp::spin(node);
   
    rclcpp::shutdown();
 
    return 0;
}