#ifndef _TOOL_PERCEPTION_H_
#define _TOOL_PERCEPTION_H_

#define LINE 128
#define CIRCLEMAXLEN 2000
#include <pcl/PCLHeader.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class pt{
public:
    int start=0;
    int end =0;
};
//单个点云对象类
class pointX
{
public:
    float x = 0;
    float y = 0;
    float z = 0;
    float r = 0;
    float d = 0;
    int type = 0; //点云的分类，0为忽略，20为障碍物，10为地面，其他的未定义
    bool isused;
    
    //附加项
    int gridx;//点在栅格中的位置
    int gridy;

    int gridxhight;//点在栅格中的位置
    int gridyhight;

    float lowest;//the lowest point in the grid
    int obj_label = 0;//聚类结果，0为未分类
    int in_map = 1;//默认在地图上为1，不在地图上为0.

    int row;//行位置
    int col;//列位置
    int smooth = 0;//光滑特性
    int label=0; //水平聚类结果

    double obj_long=0;
    double obj_wide=0;
    double obj_high=0;
};

//单个栅格对象
class CELL
{
public:
    float low;       //最低高度
    float high;      //最高高度
    float altInCept; //高度差
    int density;     //密度
    int num_2_3;
    int num_3_4;
    int num_4_5;
};

//栅格操作类
class gridFactory
{
public:
    CELL **mgrid; //点云对象
    int gw = 0;   //栅格宽度
    int gh = 0;   //栅格高度
    //栅格初始化函数
    void initGrid(int w, int h)
    {
        mgrid = new CELL *[h];
        for (int i = 0; i < h; i++)
        {
            mgrid[i] = new CELL[w];
        }
        gw = w;
        gh = h;
    }
    //栅格恢复初始状态函数
    void clearGrid()
    {
        for (int i = 0; i < gh; i++) //初始化存储激光点的栅格图
        {
            for (int j = 0; j < gw; j++)
            {
                mgrid[i][j].high = -1000;
                mgrid[i][j].low = 10000;
                mgrid[i][j].density = 0;
                mgrid[i][j].altInCept = 0;
                mgrid[i][j].num_2_3 = 0;
                mgrid[i][j].num_3_4 = 0;
                mgrid[i][j].num_4_5 = 0;
            }
        }
    }
    //栅格内部情况统计
    void gridStatistic()
    {
        std::cout << "***********************************" << std::endl;
        std::cout << "********栅格图内部情况分析*********" << std::endl;
        std::cout << "***********************************" << std::endl;
        //栅格内点数最大、最小值、平均值和分布计算
        int min = 10000;
        int max = -10000;
        double average = 0;
        int ptcount = 0;
        int distribution[10000] = {0};
        for (int y = 0; y < gh; y++)
            for (int x = 0; x < gw; x++)
            {
                if (mgrid[y][x].density > 0)
                {
                    distribution[mgrid[y][x].density]++;
                    ptcount++;
                    min = min > mgrid[y][x].density ? mgrid[y][x].density : min;
                    max = max < mgrid[y][x].density ? mgrid[y][x].density : max;
                    average += mgrid[y][x].density;
                }
            }
        average = average / ptcount;
        std::cout << "栅格最小点数" << min << std::endl;
        std::cout << "栅格最大点数" << max << std::endl;
        std::cout << "栅格平均点数" << average << std::endl;
        double variance = 0;
        for (size_t i = 1; i <= max; i++)
        {
            variance += (i - average) * (i - average) * distribution[i];
        }
        std::cout << "方差" << variance / ptcount << std::endl;
    }
    //栅格对象释放函数
    void releaseGrid()
    {
        for (size_t i = 0; i < gh; i++)
        {
            delete[] mgrid[i];
        }
        delete[] mgrid;
    }
};

//点云对象类，该类是对象存储类，所有算法只改变该类的对象值
class pointCloud
{
private:                  //外界不能访问
    pointX *mptcloutdata; //点云对象,指向点云数据区
    //点云旋转与平移，标定后写在这里就行
    double xRadian = 0.01228260+0.025;
    double yRadian = 0.02214474+0.005;
    double zRadian = -1.64800000-0.007;
    double xBias = 0;
    double yBias = 0;
    double groundheight = 2.32869589;
    int savepcdcount = 0;  //保存文件计数
    char savepcdpath[100]; //保存路径和名称
    
public:
    
    pointX **mptclout;            //点云对象
    int circlelen = CIRCLEMAXLEN; //圈点云个数，该值为浮动值
public:
    pointCloud() //内存开辟
    {
        //点云水平对齐参数
        // //需要将线的水平角度进行对齐
        // for (size_t l = 1; l < LINE; l++)
        // {
        //     int mc=0;
        //     double line1=atan2(mcloud->mptclout[0][1000].y,mcloud->mptclout[0][1000].x);
        //     double min=1000.0;
        //     for (size_t c = 800; c < 1200; c++)
        //     {
        //         double linen=atan2(mcloud->mptclout[l][c].y,mcloud->mptclout[l][c].x);
        //         if (abs(line1-linen)<min)
        //         {
        //             min=abs(line1-linen);
        //             mc=c;   
        //         }
        //     }
        //     printf("pointmove[%d]=%d;\n",l,mc-1000);
        // }
      

        mptclout = new pointX *[LINE];
        mptcloutdata = new pointX[LINE * CIRCLEMAXLEN];
        for (size_t i = 0; i < LINE; i++)
        {
            mptclout[i] = &mptcloutdata[i * CIRCLEMAXLEN];
        }
    }
    void release() //资源回收
    {
        delete[] mptclout;
        delete[] mptcloutdata;
    }

    double angle2radian(double angle) //角度转弧度
    {
        return angle / 180 * M_PI;
    }
    void coordinateCorrect(pointX &pt) //世界坐标调整
    {
        double xtemp, ytemp, ztemp;
        //绕x轴旋转弧度
        double x_Rsc[2], y_Rsc[2], z_Rsc[2];
        x_Rsc[0] = sin(xRadian);
        x_Rsc[1] = cos(xRadian);
        xtemp = pt.x;
        ytemp = pt.y * x_Rsc[1] - pt.z * x_Rsc[0];
        ztemp = pt.y * x_Rsc[0] + pt.z * x_Rsc[1];
        pt.x = xtemp;
        pt.y = ytemp;
        pt.z = ztemp;
        //绕y轴旋转弧度
        y_Rsc[0] = sin(yRadian);
        y_Rsc[1] = cos(yRadian);
        xtemp = pt.x * y_Rsc[1] + pt.z * y_Rsc[0];
        ytemp = pt.y;
        ztemp = -pt.x * y_Rsc[0] + pt.z * y_Rsc[1];
        pt.x = xtemp;
        pt.y = ytemp;
        pt.z = ztemp;
        //绕z轴旋转弧度
        z_Rsc[0] = sin(zRadian);
        z_Rsc[1] = cos(zRadian);
        xtemp = pt.x * z_Rsc[1] + pt.y * z_Rsc[0];
        ytemp = pt.y * z_Rsc[1] - pt.x * z_Rsc[0];
        ztemp = pt.z;
        pt.x = xtemp;
        pt.y = ytemp;
        pt.z = ztemp;
        //xy轴平移
        pt.x += xBias;
        pt.y += yBias;
        //地面调整
        pt.z += groundheight;
    }
    void saveAsPCD() //将文档保存为PCD
    {
        sprintf(savepcdpath, "pcd/%03d.pcd", ++savepcdcount);
        //写文件声明
        FILE *writePCDStream = fopen(savepcdpath, "wb");
        fprintf(writePCDStream, "VERSION 0.7\n");                 //版本说明
        fprintf(writePCDStream, "FIELDS x y z\n");                //维度说明
        fprintf(writePCDStream, "SIZE 4 4 4\n");                  //占用字节说明
        fprintf(writePCDStream, "TYPE F F F\n");                  //具体数据类型定义
        fprintf(writePCDStream, "WIDTH %d\n", circlelen * LINE);  //点数量
        fprintf(writePCDStream, "HEIGHT 1\n");                    //无序点云默认为1
        fprintf(writePCDStream, "POINTS %d\n", circlelen * LINE); //点数量
        fprintf(writePCDStream, "DATA ascii\n");                  //文档使用字符类型shuom
        //写点云数据
        for (size_t l = 0; l < LINE; l++)
        {
            for (int32_t c = 0; c < circlelen; c++)
            {
                fprintf(writePCDStream, "%f %f %f\n", mptclout[l][c].x, mptclout[l][c].y, mptclout[l][c].z);
            }
        }
        fclose(writePCDStream);
    }
};


class GPS
{
  public:
  double x; //纬度 latitude 
  double y; //经度 longitude
  double gpstime;
  double gpsspeed;
  double gpsdirect;
  double pitch;
  double roll;
  double yaw;
};

//跟踪框
class mybox
{
  public:
  float x;
  float y;
  float z;
  float dx;
  float dy;
  float dz;
  int pointnum;
  std::string class_label;   //--yj--//
  int closenum;//  just for tracker
  double closedis; //  just for tracker
  pcl::uint64_t timestamp;
  GPS gps;
  cv::Point2f Fpoint_A;
  cv::Point2f Fpoint_B;
  cv::Point2f Fpoint_C;
  cv::Point2f Fpoint_D;
};


#endif