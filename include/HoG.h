/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

class HoG {
public:
  HoG();

  void extractOBin(const cv::Mat* Iorient,
		   const cv::Mat* Imagn, 
		   std::vector<cv::Mat*>& out, 
		   int off);
  void calcHoGBin(const cv::Mat* IOri, 
		  const cv::Mat* IMag, 
		  std::vector<cv::Mat*>& out, 
		  int offX, 
		  int offY);
  int currentF;

private:

  void calcHoGBin(uchar* ptOrient, 
		  uchar* ptMagn, 
		  int step, 
		  double* desc);
  void binning(float v, 
	       float w, 
	       double* desc, 
	       int maxb);

  int bins;
  float binsize; 
  
  int g_w;
  //cv::Mat Gauss;
  CvMat* Gauss;

  // Gauss as vector
  float* ptGauss;
  

};

