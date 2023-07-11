#pragma once
#include <vector>
#include "tool.hpp"
// #include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
// #include <image_transport/image_transport.h>
// #include <sensor_msgs/msg/point_cloud.h>
// #include <sensor_msgs/msg/point_cloud2.h>
// #include <sensor_msgs/point_cloud_conversion.hpp>
// #include <visualization_msgs/msg/marker.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>




#define _MIN(x,y) ((x) < (y) ? (x) : (y))
#define _MAX(x,y) ((x) > (y) ? (x) : (y))

using namespace std;
using namespace cv;
using namespace Eigen;


int grid_dim_y=750;
int grid_dim_x=500;


class LineFuture{
public:
CvPoint3D32f b;
CvPoint3D32f e;
CvPoint3D32f corner;
CvPoint3D32f c;
int type;   // 0- 无法判断形状  1-I型   2-L型
int size;
};

class OneCluster{
public:
int id;
CvPoint3D32f p1;CvPoint3D32f p2;
CvPoint3D32f p3;CvPoint3D32f p4;
CvPoint3D32f pc;
double dx; double dy; double dz ; double angle;
double xmin;double xmax;double ymin; double ymax; double zmin; double zmax;
int size;
vector<cv::Point> PointIndex;
std::vector<cv::Point2f> hull;
std::string class_label;
};


class PointcloudTool
{
public:
int clusterid;
Mat obstacle_map; 
Mat MaskForCluster;

//int grid_dim_y;
//int grid_dim_x;
vector<vector<cv::Point> > contours;
vector<Vec4i>hierarchy;
Mat Struct1;
Mat Struct2;
Mat Struct3;
Mat Struct4;
vector< LineFuture > AllFuture[128];
vector<OneCluster> Clusters;
bool ISLabel[1000][1000];
bool ISNone[1000];
float min[750][500];
float max[750][500];
bool init[750][500];
public:
PointcloudTool()
{
    obstacle_map = cv::Mat::zeros(grid_dim_y,grid_dim_x,CV_8UC1);   
    clusterid=0;
    //grid_dim_y=750;
    //grid_dim_x=500;
    Struct1=getStructuringElement(MORPH_RECT,Size(1,1));
    Struct2=getStructuringElement(MORPH_RECT,Size(2,2));
    Struct3=getStructuringElement(MORPH_RECT,Size(3,3));
    MaskForCluster=cv::Mat::zeros(grid_dim_y,grid_dim_x,CV_8UC1);   
};


//obstacle_map: 250为正常障碍物点，50为初步聚类滤出的点
//mcloud->mptclout[l][c].type=20    //正常的障碍物点20;
//mcloud->mptclout[l][c].type=70;   //大面积的障碍物点
//mcloud->mptclout[l][c].type=10;   //地面点
//mcloud->mptclout[l][c].type=1;    //超出范围的点,包括高度

//============================变尺度栅格地面分割==============================
void NewGroundSeg(pointCloud *mcloud)
{
  int Col=mcloud->circlelen;
  int Row=90;
  
  obstacle_map = cv::Mat::zeros(grid_dim_y,grid_dim_x,CV_8UC1);
  //for (int x = 0; x < grid_dim_x; x++){
  //  for (int y = 0; y < grid_dim_y; y++){
  //    init[y][x]=false;
  //  }
  //}
  memset(init, 0, sizeof init);
#pragma omp parallel for
for (size_t c = 0; c < Col; c++)
{
    for (size_t l = 0; l < Row; l++)
    {
    if(!mcloud->mptclout[l][c].isused)
    continue;
    
    int grid_x= mcloud->mptclout[l][c].x*5+250;
    int grid_y= 250+mcloud->mptclout[l][c].y*5;

    mcloud->mptclout[l][c].gridx=grid_x;
    mcloud->mptclout[l][c].gridy=grid_y;
    mcloud->mptclout[l][c].row=l;
    mcloud->mptclout[l][c].col=c;

    double grid_z=mcloud->mptclout[l][c].z;

    if(grid_z>2.2)
    {
    mcloud->mptclout[l][c].type=1;
    continue;
    }

    if (grid_x > 100 && grid_x < (grid_dim_x-100) && grid_y > 100 && grid_y < (grid_dim_y-50)) {
      if (!init[grid_y][grid_x]) {
        min[grid_y][grid_x] = grid_z;
        max[grid_y][grid_x] = grid_z;
        init[grid_y][grid_x] = true;
      } else {
        min[grid_y][grid_x] = _MIN(min[grid_y][grid_x], grid_z);
        max[grid_y][grid_x] = _MAX(max[grid_y][grid_x], grid_z);
      }
     } else
       mcloud->mptclout[l][c].type=1;//超出范围的点
  } 
}
 //ROS_INFO("2");
 //==========================高度差===========================
 #pragma omp parallel for
    for (int x = 5; x < grid_dim_x-5; x++) 
    for (int y = 5; y < grid_dim_y-5; y++) 
     {
        double zmin=100000;
        for(int xx=-3;xx<3;xx++)
          for(int yy=-3;yy<3;yy++)
           if(init[y+yy][x+xx])
             zmin=zmin<min[y+yy][x+xx]?zmin:min[y+yy][x+xx];
         if((max[y][x]-zmin)>0.15 && init[y][x])
            obstacle_map.ptr<uchar>(grid_dim_y-y)[x]=250;
     }  
   FilterBasedOnObstacleMap(mcloud);
    #pragma omp parallel for
   for (size_t c = 0; c < mcloud->circlelen; c++)
     {
        for (size_t l = 0; l < Row; l++)
        {
          if(mcloud->mptclout[l][c].type==1 || !mcloud->mptclout[l][c].isused)
          continue;
          int xt= mcloud->mptclout[l][c].gridx;
          int yt= mcloud->mptclout[l][c].gridy;
          //if (xt < 0 || xt >= grid_dim_x || yt < 0 || yt >= grid_dim_y) 
          //continue;
          if(obstacle_map.ptr<uchar>(grid_dim_y-yt)[xt]==250)
          mcloud->mptclout[l][c].type=20;  //普通障碍物点
          else if(obstacle_map.ptr<uchar>(grid_dim_y-yt)[xt]==50)
          mcloud->mptclout[l][c].type=70;  //大面积的障碍物点
          else
          mcloud->mptclout[l][c].type=10;  //地面点
        }
    }
   /*
   cv::Mat show_all=obstacle_map.clone();
   for(int xt=0;xt<500;xt++)
      for(int yt=0;yt<750;yt++)
      {
        if(show_all.ptr<uchar>(yt)[xt]==0 && MaskForCluster.ptr<uchar>(yt)[xt]>0)
        show_all.ptr<uchar>(yt)[xt]=150;
      }
   */
  //  cv::imshow("showall",obstacle_map);
  //  cv::imwrite("test.png", obstacle_map);
  //  cv::waitKey(1);
}

//==========================障碍物图初步聚类过滤==============================
void FilterBasedOnObstacleMap(pointCloud *mcloud)
{
  //提取轮廓，较大的轮廓剔除，不再进行聚类处理
  contours.clear();
	hierarchy.clear();
	cv::Mat grayImage;
  Mat obstacle_map_t;
  dilate(obstacle_map,obstacle_map_t,Struct2);
	threshold(obstacle_map_t, grayImage, 0, 255,  CV_THRESH_BINARY);//CV_THRESH_BINARY |
	findContours(grayImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Scalar scolor = Scalar( 250, 250, 250);
	int xminp=1000;  int yminp=1000;
	//int xminr=1000;  int yminr=1000;
	int xmaxp=-1000; int ymaxp=-1000; 
	//int xmaxr=-1000; int ymaxr=-1000;
	int X_tem;  int Y_tem;
  Mat temp = cv::Mat::zeros(grid_dim_y,grid_dim_x,CV_8UC1);
	for (int i = 0; i < contours.size(); i++)
	{   
        if(contours[i].size()==1)
         temp.ptr<uchar>(grid_dim_y-contours[i][0].y)[contours[i][0].x]==250;
       
        xminp=1000; yminp=1000; xmaxp=-1000; ymaxp=-1000;
        
        for (int n = 0; n<contours[i].size(); n++)
		{
			Y_tem=contours[i][n].y;  X_tem=contours[i][n].x;
			ymaxp=(Y_tem>ymaxp)?Y_tem:ymaxp;
			yminp=(Y_tem<yminp)?Y_tem:yminp;
      xmaxp=(X_tem>xmaxp)?X_tem:xmaxp;
			xminp=(X_tem<xminp)?X_tem:xminp;
		}
     if((xmaxp-xminp)>40 || (ymaxp-yminp)>40 )
     drawContours(temp, contours, i, scolor, CV_FILLED);
    }

    for (int x = 5; x < grid_dim_x-5; x++) 
    for (int y = 5; y < grid_dim_y-5; y++) 
     {
        if((obstacle_map.ptr<uchar>(grid_dim_y-y)[x])>0 && (temp.ptr<uchar>(grid_dim_y-y)[x])>0)
             obstacle_map.ptr<uchar>(grid_dim_y-y)[x]=50;
     }  
};
//=========================================================================

//===============================点云聚类===================================
std::vector< std::vector< std::vector<pointX> > >  PointCloudCluster(pointCloud *mcloud)
{
    int Col=mcloud->circlelen;
    int Row=90;
    pointX temppoint;
    std::vector<std::vector< std::vector<pointX> > > allcluster;
    std::vector< std::vector<pointX> >  allclusterrow;
    std::vector<pointX>  tempcluster;
    double x,y,z,xx,yy,zz;
    for(size_t l = 0; l < Row; l++)
      {
        allclusterrow.clear(); 
        tempcluster.clear();
        //sumx=0; sumy=0;
      for (size_t c = 0; c < Col; c++)
      {
      if(!mcloud->mptclout[l][c].isused || mcloud->mptclout[l][c].type==1)
          continue;
      if(mcloud->mptclout[l][c].type!=20)
          continue;
      x= mcloud->mptclout[l][c].x;
      y= mcloud->mptclout[l][c].y;
      z= mcloud->mptclout[l][c].z;
      temppoint=mcloud->mptclout[l][c];
      if(tempcluster.size()==0)
      {
          tempcluster.push_back(temppoint);
          //sumx=sumx+x;sumy=sumy+y;
      }
      else
      { 
          xx=tempcluster[tempcluster.size()-1].x; 
          yy=tempcluster[tempcluster.size()-1].y;
          zz=tempcluster[tempcluster.size()-1].z;
          if(sqrt((x-xx)*(x-xx)+(y-yy)*(y-yy)+(z-zz)*(z-zz))<0.7)
          {
            tempcluster.push_back(temppoint);
            //sumx=sumx+x; sumy=sumy+y;
          }
          else
          {
                //tempcluster.push_back(Point(j,0));tempcluster.push_back(Point(int(sumx/(temproad.size()-1)),int(sumy/(temproad.size()-1))));
                
                int xn=tempcluster[0].x*5+250;int yn=500-tempcluster[0].y*5;
                int xm=tempcluster[tempcluster.size()-1].x*5+250;int ym=500-tempcluster[tempcluster.size()-1].y*5;
                if ((tempcluster.size()>3)||(tempcluster.size()>0 && (MaskForCluster.ptr<uchar>(ym)[xm]>0 || MaskForCluster.ptr<uchar>(yn)[xn]>0 ) ))
                allclusterrow.push_back(tempcluster);
                tempcluster.clear();  //sumx=0;  sumy=0;
                tempcluster.push_back(temppoint);
          }
      }
      }
      if (tempcluster.size() > 0) {
        int xn=tempcluster[0].x*5+250;int yn=500-tempcluster[0].y*5;
        int xm=tempcluster[tempcluster.size()-1].x*5+250;int ym=500-tempcluster[tempcluster.size()-1].y*5;
        if ((tempcluster.size()>3)||(tempcluster.size()>0 && (MaskForCluster.ptr<uchar>(ym)[xm]>0 || MaskForCluster.ptr<uchar>(yn)[xn]>0 ) ))
        {
          //tempcluster.push_back(Point(j,0));temproad.push_back(Point(int(sumx/(temproad.size()-1)),int(sumy/(temproad.size()-1))));
          allclusterrow.push_back(tempcluster);//temproad.clear();sumx=0;sumy=0;
        }
        allcluster.push_back(allclusterrow);
      }

}
return allcluster;
};
    

//=======================将每根线聚类完成的结果进行合并=========================
vector<OneCluster>  CombineClusterResult( vector< vector< vector<pointX> > > *allcluster )
{
//-----------------------计算每一个聚类的特征---------------------------------
//vector< LineFuture > AllFuture[128];
for(int k=0;k< allcluster->size();k++) 
{ AllFuture[k].clear();
  for(int n=0;n<allcluster->at(k).size();n++)
     AllFuture[k].push_back(FindLineFuture(allcluster->at(k)[n]));
}
//-------------------------------------------------------------------------
Clusters.clear();
for(int k=0; k < allcluster->size(); k++)  //128行
{
  for(int n=0;n < allcluster->at(k).size(); n++)
    { 
      if(Clusters.size()==0)
      { 
        OneCluster t=CreatOneCluster(allcluster->at(k)[n],k,n);
        Clusters.push_back(t);
        continue;
      }
         int N=FindCloseCluster(allcluster->at(k)[n],k,n);
         if(N==-1)
         {
           OneCluster t=CreatOneCluster(allcluster->at(k)[n],k,n);
           Clusters.push_back(t);
         }else{
         OneCluster temp= UpdataOneCluster(N,allcluster->at(k)[n],k,n);  
         Clusters[N]= temp;   
         }  
    }
}
//--------------------------------------------------------------------------

CalculationHull(allcluster);
//聚类结果合并,并删除较小的目标
ReClustering(allcluster);
clusterid=0;
//-------------------------------------------------------------------------
/*
Mat show= cv::Mat::zeros(750, 500,CV_8UC3);
for(int k=0;k< allcluster->size();k++)
{
  		int a=rand() % 255 + 1;
			int b=rand() % 255 + 1;
			int c=rand() % 255 + 1;
      // ROS_INFO("rrrrrrrrrr%d",allcluster->at(k).size());
      for(int n=0;n<allcluster->at(k).size();n++)
         { 
              for(int m=0;m<allcluster->at(k)[n].size();m++)
                if( allcluster->at(k)[n][m].type==20 || allcluster->at(k)[n][m].type==70 )//|| allcluster->at(k)[n][m].type==10)
                //continue;
                {
                  int xt=allcluster->at(k)[n][m].gridx; int yt=allcluster->at(k)[n][m].gridy;
                  show.at<Vec3b>(750-yt,xt)[0]=a; 
                  show.at<Vec3b>(750-yt,xt)[1]=b; 
                  show.at<Vec3b>(750-yt,xt)[2]=c;
                }
         }
}
cv::imshow("show",show); 
cv::waitKey(1);*/
return Clusters;
};
//=================================================


LineFuture FindLineFuture( vector<pointX> p)
{
  //=======================处理点太少的问题===================
  if(p.size()<=3)
   {
    p.push_back(p[p.size()-1]);
    p.push_back(p[0]);
   }
   //======================================================

   LineFuture t;
   int s=p.size();
   t.b=cvPoint3D32f(p[0].x,p[0].y,p[0].z);
   t.e=cvPoint3D32f(p[s-1].x,p[s-1].y,p[s-1].z);
   
   t.size=s;
   if (p.size()<=10)
    {
    t.c=cvPoint3D32f(0.5*(p[0].x+p[s-1].x),0.5*(p[0].y+p[s-1].y),0.5*(p[0].z+p[s-1].z));
    t.type=0;
    t.corner=cvPoint3D32f(0,0,0);
    return t;
    }
   else
   {
     double dmax=-1000;
     int num=-1;
    for(int m=1;m<s-1;m++)
    {
     double d=pointtoline(t.b,t.e,cvPoint3D32f(p[m].x,p[m].y,0));
     if(d>dmax)
     {
       dmax=d; num=m;
     }
    }
    if(dmax>0.3)
    {
      t.type=2;
      t.corner=cvPoint3D32f(p[num].x,p[num].y,p[num].z);
      t.c=cvPoint3D32f(0.3333*(t.b.x+t.e.x+t.corner.x),0.3333*(t.b.y+t.e.y+t.corner.y),0.3333*(t.b.z+t.e.z+t.corner.z));
    }else
    {
      t.type=1;
      t.corner=cvPoint3D32f(0,0,0);
      t.c=cvPoint3D32f(0.5*(p[0].x+p[s-1].x),0.5*(p[0].y+p[s-1].y),0.5*(p[0].z+p[s-1].z));
    }
   }
    return t;
};

//======================================================================
double pointtoline(CvPoint3D32f pb,CvPoint3D32f pe,CvPoint3D32f po )
{	  double a, b, c,dis;
	  // 两点式到一般式
	  // 两点式(y - y1)/(x - x1) = (y2 - y1)/(x2 - x1)
	  // 一般式(y2 - y1)x + (x1 - x2)y + (x2y1 - x1y2) = 0
	  // A = y2 - y1
	  // B = x1 - x2
	  // C = x2y1 - x1y2
	  a = pe.y - pb.y;
	  b = pb.x - pe.x;
	  c = pe.x * pb.y - pb.x * pe.y;
	  // 距离公式d = |A*x0 + B*y0 + C|/¡Ì(A^2 + B^2)
	  dis = abs(a * po.x + b * po.y + c) / sqrt(a * a + b * b);
	  return dis;

};

inline double point2point(CvPoint3D32f a,CvPoint3D32f b)
{	 
	  return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y) +(a.z-b.z)*(a.z-b.z));
};
inline double point2point(Point2f a,Point2f b)
{	 
	  return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
};
//======================================================================

OneCluster CreatOneCluster(vector<pointX> p,int n1,int n2)
{
  OneCluster t;
  t.size=p.size();
  double sumx=0;double sumy=0;
  float x_min, x_max,z_min,z_max,y_min,y_max,xx,yy,zz;
  x_min=1000;x_max=-1000;  y_min=1000;y_max=-1000;z_min=1000;z_max=-1000;
  for(int k=0;k< p.size();k++)
  { 
      xx=p[k].x;yy=p[k].y;zz=p[k].z;
      x_min=x_min<xx?x_min:xx;
      y_min=y_min<yy?y_min:yy;
      z_min=z_min<zz?z_min:zz;
      x_max=x_max>xx?x_max:xx;
      y_max=y_max>yy?y_max:yy;
      z_max=z_max>zz?z_max:zz;
    }
      t.pc=cvPoint3D32f((x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2);        
      t.xmax=x_max; t.zmax=z_max; t.ymax=y_max;
      t.xmin=x_min; t.zmin=z_min; t.ymin=y_min;
      t.dx=x_max-x_min;
      t.dy=y_max-y_min;
      t.dz=z_max-z_min; 
      
      t.p1=cvPoint3D32f(t.pc.x-t.dx/2,t.pc.y+t.dy/2,0);
      t.p2=cvPoint3D32f(t.pc.x+t.dx/2,t.pc.y+t.dy/2,0);
      t.p3=cvPoint3D32f(t.pc.x+t.dx/2,t.pc.y-t.dy/2,0);
      t.p4=cvPoint3D32f(t.pc.x-t.dx/2,t.pc.y-t.dy/2,0);

   t.PointIndex.push_back(Point(n1,n2));
   t.id=clusterid;clusterid++;
   return t;
}

//======================================================================
int FindCloseCluster(vector<pointX> p,int n1,int n2)
{
  LineFuture temp=AllFuture[n1][n2];
  int num=-1;
  double dmin=2.0;
  for(int n=0;n<Clusters.size();n++)
  {
    //double d1=point2point(temp.b,Clusters[n].pc);
    double d2=point2point(temp.e,Clusters[n].pc);
    //double d3=point2point(temp.e,Clusters[n].pc);
    //double d=d1<d2?d1:d2;
    //d=d<d3?d:d3;
    double d=d2;
    if(d<dmin)
    {
      num=n;
      dmin=d;
    }
  }
return num;
}

//=====================================================================
OneCluster UpdataOneCluster(int N,vector<pointX> p,int n1,int n2)
{
  OneCluster t=Clusters[N];
  OneCluster r;
  r.id=t.id;
  r.size=t.size+p.size();
  CvPoint3D32f pt=AllFuture[n1][n2].c;
  //r.pc=cvPoint3D32f((t.size*t.pc.x+p.size()*pt.x)/r.size,(t.size*t.pc.y+p.size()*pt.y)/r.size,(t.size*t.pc.z+p.size()*pt.z)/r.size);
  r.PointIndex=t.PointIndex;
  r.PointIndex.push_back(Point(n1,n2));

  float x_min, x_max,z_min,z_max,y_min,y_max,xx,yy,zz;
  x_min=t.xmin;x_max=t.xmax;  y_min=t.ymin;
  y_max=t.ymax;z_min=t.zmin; z_max=t.zmax;
  for(int k=0;k< p.size();k++)
  { 
      xx=p[k].x;yy=p[k].y;zz=p[k].z;
      x_min=x_min<xx?x_min:xx;
      y_min=y_min<yy?y_min:yy;
      z_min=z_min<zz?z_min:zz;
      x_max=x_max>xx?x_max:xx;
      y_max=y_max>yy?y_max:yy;
      z_max=z_max>zz?z_max:zz;
    }
      r.pc=cvPoint3D32f((x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2);        
      r.xmax=x_max; r.zmax=z_max; r.ymax=y_max;
      r.xmin=x_min; r.zmin=z_min; r.ymin=y_min;
      r.dx=x_max-x_min;
      r.dy=y_max-y_min;
      r.dz=z_max-z_min; 
      
      r.p1=cvPoint3D32f(t.pc.x-t.dx/2,t.pc.y+t.dy/2,0);
      r.p2=cvPoint3D32f(t.pc.x+t.dx/2,t.pc.y+t.dy/2,0);
      r.p3=cvPoint3D32f(t.pc.x+t.dx/2,t.pc.y-t.dy/2,0);
      r.p4=cvPoint3D32f(t.pc.x-t.dx/2,t.pc.y-t.dy/2,0);
  
  return r;
}

//====================================================================
void CalculationHull(vector< vector< vector<pointX> > > *allcluster)
{
  std::vector<cv::Point2f> points;
  for(int n=0;n<Clusters.size();n++)
  { 
    points.clear();
    for (unsigned int i = 0; i < Clusters[n].PointIndex.size(); i++)
    {
      Point Pi= Clusters[n].PointIndex[i];
      for (unsigned int j = 0; j < allcluster->at(Pi.x)[Pi.y].size(); j++)
       {
        cv::Point2f pt;
        pt.x = allcluster->at(Pi.x)[Pi.y][j].x;
        pt.y = allcluster->at(Pi.x)[Pi.y][j].y;
        points.push_back(pt);
       }
    }
   cv::convexHull(points, Clusters[n].hull);
  }
}

//====================================================================
void ReClustering(vector< vector< vector<pointX> > > *allcluster)
{
      
      if(Clusters.size()==0 || (Clusters.size()>1000))
      {
      std::cout<<"Error... SIZE:"<<Clusters.size()<<std::endl;
      return;
      }
      
      memset(ISNone, 0, sizeof ISNone);
      //memset(ISLabela, 0, sizeof ISLabela);
      //memset(ISLabelb, 0, sizeof ISLabelb);
      memset(ISLabel, 0, sizeof ISLabel);

      //for (int x = 0; x < 10000; x++)
      //   for (int y = 0; y < 10000; y++)
      //     ISLabel[y][x]=false;

      while(1)
      {
        int num=0; 
        for(int n=0;n<Clusters.size();n++)
        {
          if(ISNone[Clusters[n].id])
          continue;
        for(int m=0;m<Clusters.size();m++)
        { 
          if(ISNone[Clusters[m].id])
            continue;
          //if(Clusters[m].id>999 || Clusters[n].id>999 || Clusters[m].id<0|| Clusters[n].id<0)
          //     std::cout<<Clusters[m].id<<","<<Clusters[n].id<<std::endl;
          if((n==m)|| ISLabel[Clusters[m].id][Clusters[n].id] || (point2point(Clusters[n].pc,Clusters[m].pc)>10) )
              continue; 
            //----------------------两个边界的最短距离----------------------------------
            double dismin=100000;
            for(int k1=0;k1<Clusters[n].hull.size();k1++)
              for(int k2=0;k2<Clusters[m].hull.size();k2++)
              {
                double dtemp=point2point(Clusters[n].hull[k1],Clusters[m].hull[k2]);
                dismin=dismin<dtemp?dismin:dtemp;
              }
            //-----------------------边界是否有重叠-------------------------------
            bool IsIntersection=0;
            if(Clusters[m].hull.size()>2 && Clusters[n].hull.size()>2)
            {
              for(int k=0;k<Clusters[m].hull.size();k++)
                if (!(pointPolygonTest(Clusters[n].hull, Clusters[m].hull[k], 1)<0))
                {
                  IsIntersection=1;
                  break;
                }
                if(IsIntersection==0){
                for(int k=0;k<Clusters[n].hull.size();k++)
                if (!(pointPolygonTest(Clusters[m].hull, Clusters[n].hull[k], 1)<0))
                  {
                    IsIntersection=1;
                    break;
                  }
               }
            }
            //---------------------------------------------------------------------
            bool IsMask=0;//是否属于同一个mask;
            int xn=Clusters[n].pc.x*5+250;int yn=500-Clusters[n].pc.y*5;
            int xm=Clusters[m].pc.x*5+250;int ym=500-Clusters[m].pc.y*5;
            int pix1=-1;int pix2=1;
            if(ym>0 && ym<750 && xm>0 && xm<500)
            pix1=MaskForCluster.ptr<uchar>(ym)[xm];
            if(yn>0 && yn<750 && xn>0 && xn<500)
            pix2=MaskForCluster.ptr<uchar>(yn)[xn];
            IsMask=((pix1==pix2)&&pix1!=0&&pix2!=0);
            //---------------------------------------------------------------------------
            if (dismin<0.3 || IsIntersection || IsMask) //
            {
                  ISNone[Clusters[m].id]=true;ISNone[Clusters[n].id]=true;
                  if(n<m)
                  MergeClusterResult(n,m,allcluster);
                  else
                  MergeClusterResult(m,n,allcluster);
                  num++;
                  break;
            } else {;
                //if(Clusters[m].id>999 || Clusters[n].id>999 || Clusters[m].id<0|| Clusters[n].id<0)
                //std::cout<<Clusters[m].id<<","<<Clusters[n].id<<std::endl;
                ISLabel[Clusters[m].id][Clusters[n].id]=true;
            }
        }
        if(num>0)
        break;
        } 
        if(num==0)
        break;
      }

    //剔除不满足要求的目标
    /*int kk=0;
    while(1)
    { bool flag=0;
      if(kk<Clusters.size())
      for(int n=kk;n<Clusters.size();n++)
      if((Clusters[n].dx<0.9 && Clusters[n].dy<0.9 && Clusters[n].dz<0.9 )|| Clusters[n].zmax>2.7 || Clusters[n].size<20)
      {
      std::vector<OneCluster>::iterator it = Clusters.begin();
      Clusters.erase(it+n); flag=1;
      kk=n;
      break;
      }
      if(!flag)
      break;
    }*/
    vector<OneCluster> temp_Clusters;
      for (size_t n = 0; n < Clusters.size(); n++)
      {
        if (ISNone[Clusters[n].id]==0  && ! ((Clusters[n].dx<0.9 && Clusters[n].dy<0.9 && Clusters[n].dz<0.9 )|| Clusters[n].zmax>2.7 || Clusters[n].size<20))
        {
          temp_Clusters.push_back(Clusters[n]);
        }
      }
      std::vector<OneCluster>().swap(Clusters);
      Clusters = temp_Clusters;
}
//====================================================================
//合并两个聚类结果
void MergeClusterResult(int n1,int n2,vector< vector< vector<pointX> > > *allcluster)
{
  OneCluster t1=Clusters[n1];
  OneCluster t2=Clusters[n2];
  OneCluster r;
  r.size=t1.size+t2.size;

  r.PointIndex=t1.PointIndex;
  r.PointIndex.insert(r.PointIndex.end(),t2.PointIndex.begin(),t2.PointIndex.end());

  //CvPoint3D32f pt=AllFuture[n1][n2].c;
  float x_min, x_max,z_min,z_max,y_min,y_max;
  x_min=t1.xmin;x_max=t1.xmax;  y_min=t1.ymin;
  y_max=t1.ymax;z_min=t1.zmin; z_max=t1.zmax;

  x_min=x_min<t2.xmin?x_min:t2.xmin;
  y_min=y_min<t2.ymin?y_min:t2.ymin;
  z_min=z_min<t2.zmin?z_min:t2.zmin;
  x_max=x_max>t2.xmax?x_max:t2.xmax;
  y_max=y_max>t2.ymax?y_max:t2.ymax;
  z_max=z_max>t2.zmax?z_max:t2.zmax;

  r.pc=cvPoint3D32f((x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2);        
  r.xmax=x_max; r.zmax=z_max; r.ymax=y_max;
  r.xmin=x_min; r.zmin=z_min; r.ymin=y_min;
  r.dx=x_max-x_min;
  r.dy=y_max-y_min;
  r.dz=z_max-z_min; 

  r.p1=cvPoint3D32f(r.pc.x-r.dx/2,r.pc.y+r.dy/2,0);
  r.p2=cvPoint3D32f(r.pc.x+r.dx/2,r.pc.y+r.dy/2,0);
  r.p3=cvPoint3D32f(r.pc.x+r.dx/2,r.pc.y-r.dy/2,0);
  r.p4=cvPoint3D32f(r.pc.x-r.dx/2,r.pc.y-r.dy/2,0);

  std::vector<cv::Point2f> points;
    points.clear();
    for (unsigned int i = 0; i < r.PointIndex.size(); i++)
    {
      Point Pi= r.PointIndex[i];
      for (unsigned int j = 0; j < allcluster->at(Pi.x)[Pi.y].size(); j++)
      {
        cv::Point2f pt;
        pt.x = allcluster->at(Pi.x)[Pi.y][j].x;
        pt.y = allcluster->at(Pi.x)[Pi.y][j].y;
        points.push_back(pt);
      }
    }
   /* for (unsigned int j = 0; j < Clusters[n1].hull.size(); j++)
      {
        cv::Point2f pt;
        pt.x = Clusters[n1].hull[j].x;
        pt.y = Clusters[n1].hull[j].y;
        points.push_back(pt);
      }
    for (unsigned int j = 0; j < Clusters[n2].hull.size(); j++)
      {
        cv::Point2f pt;
        pt.x = Clusters[n2].hull[j].x;
        pt.y = Clusters[n2].hull[j].y;
        points.push_back(pt);
      }*/
  cv::convexHull(points, r.hull);
  r.id=clusterid;clusterid++;
  Clusters.push_back(r);
  
  /*std::vector<OneCluster>::iterator it = Clusters.begin();
  Clusters.erase(it+n1);
  it = Clusters.begin() + n2-1;
  Clusters.erase(it);
  std::vector<OneCluster> temp=Clusters;
  Clusters.swap(temp); */ //释放内存
}
//====================================================================
// void ShowResultWithBox(visualization_msgs::msg::Marker &points,visualization_msgs::msg::Marker &line_list)
// {
//   points.header.frame_id = line_list.header.frame_id = "PERCEPTION2023";
//   points.header.stamp =line_list.header.stamp = rclcpp::Clock(RCL_ROS_TIME).now();
//   points.ns = "points";
//   line_list.ns = "points_and_lines";
//   points.action =line_list.action = visualization_msgs::msg::Marker::ADD;
//   points.pose.orientation.w = line_list.pose.orientation.w = 1.0;
//   points.id = 0;
//   line_list.id = 2;
//   points.type = visualization_msgs::msg::Marker::POINTS;
//   line_list.type = visualization_msgs::msg::Marker::LINE_LIST;
//   points.scale.x = 0.5;points.scale.y = 0.5;points.scale.z = 0.5;
//   line_list.scale.x = 0.1;// Points are green
//   points.color.g = 1.0f; points.color.a = 1.0;// Line list is red
//   line_list.color.r = 1.0;line_list.color.g = 1.0;line_list.color.b = 1.0;line_list.color.a = 1.0;
//   geometry_msgs::msg::Point p1,p2,p3,p4,p5,p6,p7,p8, p0;
//   float max_z,min_z;Point2f pot1,pot2,pot3,pot4;
   
//    for(int n=0;n<Clusters.size();n++)
//    {
//         pot1=Point2f(Clusters[n].p1.x,Clusters[n].p1.y);
//         pot2=Point2f(Clusters[n].p2.x,Clusters[n].p2.y);
//         pot3=Point2f(Clusters[n].p3.x,Clusters[n].p3.y);
//         pot4=Point2f(Clusters[n].p4.x,Clusters[n].p4.y);
//         max_z=Clusters[n].zmax;
//         min_z=Clusters[n].zmin;
        
//         p0.x=0.25*(pot1.x+pot2.x+pot3.x+pot4.x);
//         p0.y=0.25*(pot1.y+pot2.y+pot3.y+pot4.y); 
//         p0.z=0.5*(max_z+min_z);

//         p1.x = pot1.x;p1.y = pot1.y;p1.z = min_z;
//         p2.x=pot2.x;p2.y=pot2.y;p2.z=min_z;
//         p3.x=pot3.x;p3.y=pot3.y;p3.z=min_z;
//         p4.x=pot4.x; p4.y=pot4.y;p4.z=min_z;
//         p5.x = pot1.x;p5.y = pot1.y;p5.z = max_z;
//         p6.x=pot2.x;p6.y=pot2.y;p6.z=max_z;
//         p7.x=pot3.x;p7.y=pot3.y;p7.z=max_z;
//         p8.x=pot4.x;p8.y=pot4.y;p8.z=max_z;

//         points.points.push_back(p0);
//         line_list.points.push_back(p1);line_list.points.push_back(p2);
//         line_list.points.push_back(p2); line_list.points.push_back(p3);
//         line_list.points.push_back(p3);line_list.points.push_back(p4);
//         line_list.points.push_back(p4);line_list.points.push_back(p1);
//         line_list.points.push_back(p5); line_list.points.push_back(p6);
//         line_list.points.push_back(p6); line_list.points.push_back(p7);
//         line_list.points.push_back(p7); line_list.points.push_back(p8);
//         line_list.points.push_back(p8);line_list.points.push_back(p5);
//         line_list.points.push_back(p1); line_list.points.push_back(p5);
//         line_list.points.push_back(p2);line_list.points.push_back(p6);
//         line_list.points.push_back(p3); line_list.points.push_back(p7);
//         line_list.points.push_back(p4); line_list.points.push_back(p8);
//    }

// }

//====================================================================
// void ShowResultWithHull(visualization_msgs::msg::Marker &points,visualization_msgs::msg::Marker &line_list)
// {
//   points.header.frame_id = line_list.header.frame_id = "PERCEPTION2023";
//   points.header.stamp =line_list.header.stamp =  rclcpp::Clock(RCL_ROS_TIME).now();
//   points.ns = "points";
//   line_list.ns = "points_and_lines";
//   points.action =line_list.action = visualization_msgs::msg::Marker::ADD;
//   points.pose.orientation.w = line_list.pose.orientation.w = 1.0;
//   points.id = 0;
//   line_list.id = 2;
//   points.type = visualization_msgs::msg::Marker::POINTS;
//   line_list.type = visualization_msgs::msg::Marker::LINE_LIST;
//   points.scale.x = 0.5;points.scale.y = 0.5;points.scale.z = 0.5;
//   line_list.scale.x = 0.1;// Points are green
//   points.color.g = 1.0f; points.color.a = 1.0;// Line list is red
//   line_list.color.r = 1.0;line_list.color.g = 1.0;line_list.color.b = 1.0;line_list.color.a = 1.0;
//   geometry_msgs::msg::Point p1,p2,p3,p4,p5,p6,p7,p8, p0;
//   float max_z,min_z;Point2f pot1,pot2,pot3,pot4;
   
//   for(int n=0;n<Clusters.size();n++)
//   {
//       max_z=Clusters[n].zmax;
//       min_z=Clusters[n].zmin;
//       p0.x=Clusters[n].pc.x; p0.y=Clusters[n].pc.y; p0.z=0.5*(max_z+min_z);
//       points.points.push_back(p0);
      
//       for(int m=0;m<Clusters[n].hull.size()-1;m++)
//       {
//         p1.x = Clusters[n].hull[m].x;p1.y = Clusters[n].hull[m].y;  p1.z = min_z;
//         p2.x=Clusters[n].hull[m+1].x; p2.y=Clusters[n].hull[m+1].y;  p2.z=min_z;
//         p3.x=p1.x;p3.y=p1.y;p3.z=max_z;
//         p4.x=p2.x;p4.y=p2.y;p4.z=max_z;
//         line_list.points.push_back(p1);line_list.points.push_back(p2);
//         line_list.points.push_back(p3); line_list.points.push_back(p4);
//         line_list.points.push_back(p1);line_list.points.push_back(p3);
//       }

//         p1.x = Clusters[n].hull[Clusters[n].hull.size()-1].x;p1.y = Clusters[n].hull[Clusters[n].hull.size()-1].y;  p1.z = min_z;
//         p2.x=Clusters[n].hull[0].x; p2.y=Clusters[n].hull[0].y;  p2.z=min_z;
//         p3.x=p1.x;p3.y=p1.y;p3.z=max_z;
//         p4.x=p2.x;p4.y=p2.y;p4.z=max_z;
//         line_list.points.push_back(p1);line_list.points.push_back(p2);
//         line_list.points.push_back(p3); line_list.points.push_back(p4);
//         line_list.points.push_back(p1);line_list.points.push_back(p3);
//    }
// }

};//类