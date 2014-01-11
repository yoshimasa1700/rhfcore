#ifndef __CRFOREST__
#define __CRFOREST__

#include "CRTree.h"
#include "CPatch.h"
#include "util.h"
#include "HoG.h"
#include "CDataset.h"
//#include <memory>
#include <boost/shared_ptr.hpp>

#include "CDetectionResult.h"
//#include <boost/timer/timer.hpp>

#ifndef __APPLE__
#include <omp.h>
#endif

cv::Mat calcGaussian(double score, double center);

class paramBin {
 public:
  //paramBin(){
  //    this->next = NULL;
  //}
  paramBin(double conf, double r, double p, double y){
    setParam(r, p, y);
    confidence = conf;
  }

  ~paramBin(){}

  //paramBin& operator+(const paramBin&);
  //paramBin& operator+=(const paramBin&);

  boost::shared_ptr<paramBin> next;
  void setParam(double r,double p, double y){
    roll = r; 
    pitch = p; 
    yaw = y;
  }

  void addChild(double conf, double r, double p, double y){
    if(!next)
      next = boost::shared_ptr<paramBin>(new paramBin(conf, r, p, y));
    else
      next->addChild(conf, r, p, y);
  }
    
  double roll, pitch, yaw;
  double confidence;
};

static HoG hog;

class CRForest {
 public:
  CRForest(CConfig config){
    conf = config;
    vTrees.resize(conf.ntrees);
  }
  ~CRForest() {
    int numberOfTrees = vTrees.size();
    for(int i = 0;i < numberOfTrees;++i){
      if(vTrees.at(i) != NULL)
	delete vTrees.at(i);
    }
  }

  void learning();

  void growATree(const int treeNum);

  CDetectionResult detection(CTestDataset &testSet) const;//, std::vector<double> &detectionResult, int &detectClass) const;

  void extractPatches(std::vector<std::vector<CPatch> > &patches,
		      const std::vector<CDataset> dataSet,
		      const cv::vector<cv::vector<cv::Mat*> > &image,
		      /*boost::mt19937 gen, */CConfig conf);

  void extractPatches(std::vector<std::vector<CPatch> > &patches,
		      const std::vector<CDataset> dataSet,
		      const cv::vector<cv::vector<cv::Mat*> > &image,
		      const cv::vector<cv::vector<cv::Mat*> > &negImage,
		      CConfig conf,
		      const int treeNum);

  void loadForest();

  // Regression
  void regression(std::vector<const LeafNode*>& result,
		  CTestPatch &patch) const;



  //  void loadImages(cv::vector<cv::vector<cv::Mat*> > &img,
  //		  std::vector<CDataset> &dataSet);

  //  void extractFeatureChannels(const cv::Mat* img,
  //			      cv::vector<cv::Mat*>& vImg) const;
  //  void minFilter(cv::Mat* src, cv::Mat* des, int fWind) const;
  //  void maxFilter(cv::Mat* src, cv::Mat* des, int fWind) const;

  //void voteResult(int classNumber, )
    
  CClassDatabase classDatabase;
  double matrix[16];
  double matrixI[16];

 private:
  CConfig		conf;
  std::vector<CRTree*>	vTrees;
  //CGlObjLoader *obj;
};

//inline void CRForest::extractFeatureChannels(const cv::Mat* img, cv::vector<cv::Mat*>& vImg) const{
//  vImg.clear();
//  vImg.resize(32);
//  for(int i = 0; i < 32; ++i)
//    vImg.at(i) = new cv::Mat(img->rows, img->cols, CV_8UC1);


//  //std::cout << img->channels() << std::endl;

//  cv::cvtColor(*img, *(vImg.at(0)), CV_RGB2GRAY);


//  cv::Mat I_x(img->rows, img->cols, CV_16SC1);
//  cv::Mat I_y(img->rows, img->cols, CV_16SC1);


//  cv::Sobel(*(vImg.at(0)), I_x, CV_16S, 1, 0);
//  cv::Sobel(*(vImg.at(0)), I_y, CV_16S, 0, 1);

//  cv::convertScaleAbs(I_x, *(vImg[3]), 0.25);
//  cv::convertScaleAbs(I_y, *(vImg[4]), 0.25);

//  //std::cout << "vimg[3]" << std::endl;

//   /* cv::namedWindow("test"); */
//   /* cv::imshow("test",*(vImg[3])); */
//   /* cv::waitKey(0); */
//   /* cv::destroyWindow("test"); */

//  // Orientation of gradients
//  for(int  y = 0; y < img->rows; y++)
//    for(int  x = 0; x < img->cols; x++) {
//      // Avoid division by zero
//      float tx = (float)I_x.at<short>(y, x) + (float)copysign(0.000001f, I_x.at<short>(y, x));
//      // Scaling [-pi/2 pi/2] -> [0 80*pi]
//      vImg.at(1)->at<uchar>(y, x) = (uchar)(( atan((float)I_y.at<short>(y, x) / tx) + 3.14159265f / 2.0f ) * 80);
//      //std::cout << "scaling" << std::endl;
//      vImg.at(2)->at<uchar>(y, x) = (uchar)sqrt((float)I_x.at<short>(y, x)* (float)I_x.at<short>(y, x) + (float)I_y.at<short>(y, x) * (float)I_y.at<short>(y, x));
//    }

//  // Magunitude of gradients
//  for(int y = 0; y < img->rows; y++)
//      for(int x = 0; x < img->cols; x++ ) {
//	vImg.at(2)->at<uchar>(y, x) = (uchar)sqrt(I_x.at<short>(y, x)*I_x.at<short>(y, x) + I_y.at<short>(y, x) * I_y.at<short>(y, x));
//      }

//  hog.extractOBin(vImg[1], vImg[2], vImg, 7);

//  // calc I_xx I_yy
//  cv::Sobel(*(vImg.at(0)), I_x, CV_16S, 2, 0);
//  cv::Sobel(*(vImg.at(0)), I_y, CV_16S, 0, 2);

//  cv::convertScaleAbs(I_x, *(vImg[5]), 0.25);
//  cv::convertScaleAbs(I_y, *(vImg[6]), 0.25);

//  cv::Mat img_Lab;
//  cv::cvtColor(*img, img_Lab, CV_RGB2Lab);
//  cv::vector<cv::Mat> tempVImg(3);

//  cv::split(img_Lab, tempVImg);

//  for(int i = 0; i < 3; ++i)
//    tempVImg.at(i).copyTo(*(vImg.at(i)));

//  // min filter
//  for(int c = 0; c < 16; ++c)
//    minFilter(vImg[c], vImg[c + 16], 5);

////  for(int i = 0; i < 32; ++i){
////      cv::namedWindow("test");
////      cv::imshow("test",*vImg.at(i));
////      cv::waitKey(0);
////      cv::destroyWindow("test");
////  }

//  for(int c = 0; c < 16; ++c)
//    maxFilter(vImg[c], vImg[c], 5);



//  /* std::cout << "extructing feature 2 " << std::endl; */

//}

//inline void CRForest::minFilter(cv::Mat* src, cv::Mat* des, int fWind) const{
//    int d = (fWind - 1) / 2;
//    cv::Rect roi;
//    cv::Mat desTemp(src->rows, src->cols, CV_8U), vTemp;

//    for(int y = 0; y < src->rows - fWind; ++y){ //for image height
//        if(y < fWind)
//            roi = cv::Rect(0, 0, src->cols, fWind - y);
//        else
//            roi = cv::Rect(0, y, src->cols, fWind);

//        cv::reduce((*src)(roi), vTemp, 0, CV_REDUCE_MIN);

//        roi = cv::Rect(0, y + d, src->cols, 1);
//        //cv::Mat roiDesTemp(desTemp, roi);
//        vTemp.copyTo(desTemp(roi));
//    }// For image height

//    for(int x = 0; x < src->cols - fWind; ++x){ // for image width
//        if(x < d)
//            roi = cv::Rect(0, 0, fWind - x, src->rows);
//        else
//            roi = cv::Rect(x, 0, fWind, src->rows);

//        cv::reduce(desTemp(roi), vTemp, 1, CV_REDUCE_MIN);

//        roi = cv::Rect(x + d, 0, 1, src->rows);
//        cv::Mat roiDesTemp((*des), roi);
//        vTemp.copyTo((*des)(roi));// = vTemp.clone();//copyTo((*des)(roi));
//    } // for image width
//}

//inline void CRForest::maxFilter(cv::Mat* src, cv::Mat* des, int fWind) const{
//  int d = (fWind - 1) / 2;
//  cv::Rect roi;
//  cv::Mat desTemp(src->rows, src->cols, CV_8U), vTemp;

//    for(int y = 0; y < src->rows - fWind; ++y){ //for image height
//      if(y < fWind)
//	roi = cv::Rect(0, 0, src->cols, fWind - y);
//      else
//	roi = cv::Rect(0, y, src->cols, fWind);

//      cv::reduce((*src)(roi), vTemp, 0, CV_REDUCE_MAX);

//      roi = cv::Rect(0, y + d, src->cols, 1);
//      cv::Mat roiDesTemp(desTemp, roi);
//      vTemp.copyTo(desTemp(roi));
//    }// For image height

//  for(int x = 0; x < src->cols - fWind; ++x){ // for image width
//    if(x < d)
//      roi = cv::Rect(0, 0, fWind - x, src->rows);
//    else
//      roi = cv::Rect(x, 0, fWind, src->rows);

//    cv::reduce(desTemp(roi), vTemp, 1, CV_REDUCE_MAX);

//    roi = cv::Rect(x + d, 0, 1, src->rows);
//    cv::Mat roiDesTemp(*des, roi);
//    vTemp.copyTo((*des)(roi));// = vTemp.clone();//copyTo((*des)(roi));
//  } // for image width


//}

#endif
