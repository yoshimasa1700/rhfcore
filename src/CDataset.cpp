#include "../include/CDataset.h"

CParamset& CParamset::operator+=(const CParamset& obj){
  //  this->setCenterPoint(this->getCenterPoint() + obj.getCenterPoint());
  //this->setRelativePosition(this->getRelativePosition() + obj.getRelativePosition());
  double tempAngle[3];
  for(int i = 0; i < 3; ++i)
    tempAngle[i] = this->getAngle()[i] + obj.getAngle()[i];
  this->setAngle(tempAngle);

  return *this;
}

CParamset& CParamset::operator/=(const float& div){
  //this->setCenterPoint(cv::Point_<double>(this->getCenterPoint().x / (int)div, this->getCenterPoint().y / (int)div));
  double tempAngle[3];
  for(int i = 0; i < 3; ++i)
    tempAngle[i] = this->getAngle()[i] / div;
  this->setAngle(tempAngle);

  return *this;
}

std::string CParamset::outputParam(){
  std::stringstream sstream;

  sstream << //this->className << " " <<
    this->centerPoint.x << " " << this->centerPoint.y << " "<<
    this->angle[0] << " " <<
    this->angle[1] << " " <<
    this->angle[2] << " ";
  return sstream.str();

}

void cropImageAndDepth(cv::Mat* rgb, cv::Mat* depth, double mindist, double maxdist){
  cv::Mat depthForView = cv::Mat(depth->rows, depth->cols, CV_8U);


  

  if(depth->type() != CV_16U){
    std::cout << "error! input depth image is wrong type!!!" << std::endl;
    exit(-1);
  }
  //  std::cout << maxdist << std::endl;
  cv::Mat allMaxDepth = cv::Mat::ones(depth->rows, depth->cols, CV_16U) * (ushort)maxdist;

  cv::Mat allMaxDepthOne = cv::Mat::ones(depth->rows, depth->cols, CV_16U) * (ushort)(maxdist-1);

  cv::min(*depth, allMaxDepth, *depth);
  cv::bitwise_and(allMaxDepthOne, *depth, *depth);
}

CDataset::CDataset()
  :imgFlag(0),
   featureFlag(0)
{
  img.clear();
  feature.clear();
  // rgb = "NULL";
  // depth = "NULL";
  // mask = "NULL";
}

CDataset::~CDataset(){
  if(imgFlag){
    releaseImage();
    //std::cout << "image released!" << std::endl;
  }

  if(featureFlag){
    releaseFeatures();
    //std::cout << "feature released!" << std::endl;
  }
}

int CDataset::loadImage(const CConfig &conf){
  cv::Mat *rgbImg, *depthImg;

  if(!conf.demoMode){
    rgbImg = new cv::Mat;
    cv::imread(rgb,3).copyTo(*rgbImg);
    if(rgbImg->data == NULL){
      std::cout << "error! rgb image file " << rgb << " not found!" << std::endl;
      exit(-1);
    }else if(rgbImg->channels() == 1){
      std::cout << "error! invarid rgb image please use 3 channel color image" << std::endl;
    }
    

    // cv::Mat maskImage = cv::imread(mask);
    // if(maskImage.data != NULL){
    //   maskImage /= 255;
    //   *rgbImg  = rgbImg->mul(maskImage);
    // }

    depthImg = new cv::Mat;
    cv::imread(depth, CV_LOAD_IMAGE_ANYDEPTH).copyTo(*depthImg);

    if(depthImg->data == NULL){
      if(conf.learningMode == 2){

	img.push_back(rgbImg);
	imgFlag  = 1;
	return -1;
      }else{
	std::cout << "error! depth image file " << depth << " not found!" << std::endl;
	exit(-1);
      }
    }else{
      //      cv::GaussianBlur(*depthImg,*depthImg, cv::Size(21,21),0);
      // if(!this->mask.empty()){
      //   cv::Mat maskImage = cv::imread(mask,0);
      //   maskImage.convertTo(maskImage,CV_16U,1.0/255.0);
      //     //	maskImage /= 255;
      //   *depthImg  = depthImg->mul(maskImage);
      // }
    }
    if(conf.learningMode != 2)
      cropImageAndDepth(rgbImg, depthImg, conf.mindist, conf.maxdist);

    img.push_back(rgbImg);
    img.push_back(depthImg);
  }else{
    if(conf.learningMode != 2)
      cropImageAndDepth(img[0], img[1], conf.mindist, conf.maxdist);
  }

  //std::cout << "depth image " << depthImg->rows << " " << depthImg->cols << std::endl;
  //cv::Mat showDepth = cv::Mat(depthImg->rows, depthImg->cols, CV_8U);
  //depthImg->convertTo(showDepth, CV_8U, 255.0 / 1000.0);

  // cv::namedWindow("test");
  // cv::imshow("test", *rgbImg);
  // cv::namedWindow("test2");
  // cv::imshow("test2", showDepth);

  // cv::waitKey(0);

  // cv::destroyWindow("test");
  // cv::destroyWindow("test2");

  imgFlag  = 1;

  return 0;
}

int CDataset::releaseImage(){
  if(imgFlag == 0){
    std::cout << "image is already released! foolish!" << std::endl;
    return -1;
  }

  for(unsigned int i = 0; i < img.size(); ++i){
    delete img[i];
  }

  imgFlag = 0;

  return 0;
}

int CParamset::showParam(){
  std::cout << "name : " << this->className;
  std::cout << " center point : " << this->centerPoint << std::endl;
  std::cout << " Pose : roll " << this->angle[0];
  std::cout << " pitch " << this->angle[1];
  std::cout << " yaw " << this->angle[2];


  std::cout << std::endl;
  return 0;
}

int CDataset::calcHoG(int type){
  int imgRow = this->img[type]->rows, 
    imgCol = this->img[type]->cols;

  int currentF = feature.size();
  this->hog.currentF = currentF;

  feature.resize(currentF + 32);
  for(int i = currentF; i < currentF + 32; ++i)
    feature[currentF + i] = new cv::Mat(imgRow, imgCol, CV_8UC1);

  cv::cvtColor(*img[type], *(feature[currentF]), CV_RGB2GRAY);

  cv::Mat I_x(imgRow, imgCol, CV_16SC1);
  cv::Mat I_y(imgRow, imgCol, CV_16SC1);


  cv::Sobel(*(feature[currentF]), I_x, CV_16S, 1, 0);
  cv::Sobel(*(feature[currentF]), I_y, CV_16S, 0, 1);

  cv::convertScaleAbs(I_x, *(feature[currentF + 3]), 0.25);
  cv::convertScaleAbs(I_y, *(feature[currentF + 4]), 0.25);

  // Orientation of gradients
  for(int  y = 0; y < img[type]->rows; y++)
    for(int  x = 0; x < img[type]->cols; x++) {
      // Avoid division by zero
      float tx = (float)I_x.at<short>(y, x) + (float)copysign(0.000001f, I_x.at<short>(y, x));
      // Scaling [-pi/2 pi/2] -> [0 80*pi]
      feature[currentF + 1]->at<uchar>(y, x) = (uchar)(( atan((float)I_y.at<short>(y, x) / tx) + 3.14159265f / 2.0f ) * 80);
      //std::cout << "scaling" << std::endl;
      feature[currentF + 2]->at<uchar>(y, x) = (uchar)sqrt((float)I_x.at<short>(y, x)* (float)I_x.at<short>(y, x) + (float)I_y.at<short>(y, x) * (float)I_y.at<short>(y, x));
    }

  // Magunitude of gradients
  for(int y = 0; y < img[type]->rows; y++)
    for(int x = 0; x < img[type]->cols; x++ ) {
      feature[currentF + 2]->at<uchar>(y, x) = (uchar)sqrt(I_x.at<short>(y, x)*I_x.at<short>(y, x) + I_y.at<short>(y, x) * I_y.at<short>(y, x));
    }

  hog.extractOBin(feature[currentF + 1], feature[currentF + 2], feature, 7);

  // calc I_xx I_yy
  cv::Sobel(*(feature[currentF]), I_x, CV_16S, 2, 0);
  cv::Sobel(*(feature[currentF]), I_y, CV_16S, 0, 2);

  cv::convertScaleAbs(I_x, *(feature[currentF + 5]), 0.25);
  cv::convertScaleAbs(I_y, *(feature[currentF + 6]), 0.25);

  // cv::Mat img_Lab;
  // cv::cvtColor(*img[type], img_Lab, CV_RGB2Lab);
  // cv::vector<cv::Mat> tempfeature(3);

  // cv::split(img_Lab, tempfeature);

  // for(int i = 0; i < 3; ++i)
  //   tempfeature[i].copyTo(*(feature[i]));

  // min max filter
  for(int c = 0; c < 16; ++c)
    minFilter(feature[currentF + c], feature[currentF + c + 16], 5);
  for(int c = 0; c < 16; ++c)
    maxFilter(feature[currentF + c], feature[currentF + c], 5);
}

int CDataset::extractFeatures(const CConfig& conf){

  int imgRow = this->img[0]->rows, 
    imgCol = this->img[0]->cols;
  cv::Mat *integralMat;
  feature.clear();


  switch(conf.learningMode){
  case 0: //rgbd
    // calc rgb feature
    switch(conf.rgbFeature){
    case 0:
      
      break;
    case 1:
      calcHoG(0);
      break;
    case 2:
      {
	// calc gray integral image
	cv::Mat grayImg(imgRow, imgCol, CV_8U);
	cv::cvtColor(*img[0], grayImg, CV_RGB2GRAY);
	integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
	cv::integral(grayImg, *integralMat, CV_64F);
	feature.push_back(integralMat);



	// calc r g b integral image
	std::vector<cv::Mat> splittedRgb(0);
	cv::split(*img[0], splittedRgb);
	for(unsigned int i = 0; i < splittedRgb.size(); ++i){
	  integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
	  cv::integral(splittedRgb[i], *integralMat, CV_64F);
	  feature.push_back(integralMat);
	}
      }
      break;
    default:
      std::cout << "undefined rgb feature, check config.xml" << std::endl;
      break;

    }
    
    // calc depth feature
    switch(conf.depthFeature){
    case 0:
      break;
    case 1:
      calcHoG(1);
      break;
    case 2:
      {
	cv::Mat tempDepth = cv::Mat(img[0]->rows, img[0]->cols, CV_8U);// = *img.at(1);
	if(img[1]->type() != CV_8U)
	  img[1]->convertTo(tempDepth, CV_8U, 255.0 / (double)(conf.maxdist - conf.mindist));
	else
	  tempDepth = *img[1];
	integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
	cv::integral(tempDepth, *integralMat, CV_64F);
	feature.push_back(integralMat);
      }
      break;
    default:
      std::cout << "undefined depth feature, check config.xml" << std::endl;
      break;
    }
    break;
    
  case 1: //depth only
    break;
  case 2: //rgb only
     // calc rgb feature
    switch(conf.rgbFeature){
    case 0:
      
      break;
    case 1:
      calcHoG(0);
      break;
    case 2:
      {
	// calc gray integral image
	cv::Mat grayImg(imgRow, imgCol, CV_8U);
	cv::cvtColor(*img[0], grayImg, CV_RGB2GRAY);
	integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
	cv::integral(grayImg, *integralMat, CV_64F);
	feature.push_back(integralMat);


        // std::cout << integralMat->at<double>(imgRow,imgCol) << std::endl;

        // cv::namedWindow("tetete");
        // cv::imshow("tetete", *feature[0]);
        // cv::waitKey(0);
        
	// calc r g b integral image
	std::vector<cv::Mat> splittedRgb(0);
	cv::split(*img[0], splittedRgb);
	for(unsigned int i = 0; i < splittedRgb.size(); ++i){
	  integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
	  cv::integral(splittedRgb[i], *integralMat, CV_64F);
	  feature.push_back(integralMat);
	}
      }
      break;
    default:
      std::cout << "undefined rgb feature, check config.xml" << std::endl;
      break;
    }
    break;
  default:
    break;
  }

  this->featureFlag = 1;
  //learning mode 1:depth 2:rgb 0:rgbd
  //feature  haar-like : 0, HOG : 1, rotated haar-like : 2
  // if(conf.learningMode != 1){
  //   if(conf.rgbFeature == 1){ // if got rgb image only, calc hog feature
  //     feature.clear();
  //     feature.resize(32);
  //     for(int i = 0; i < 32; ++i)
  // 	feature[i] = new cv::Mat(imgRow, imgCol, CV_8UC1);

  //     cv::cvtColor(*img[0], *(feature[0]), CV_RGB2GRAY);

  //     cv::Mat I_x(imgRow, imgCol, CV_16SC1);
  //     cv::Mat I_y(imgRow, imgCol, CV_16SC1);


  //     cv::Sobel(*(feature[0]), I_x, CV_16S, 1, 0);
  //     cv::Sobel(*(feature[0]), I_y, CV_16S, 0, 1);

  //     cv::convertScaleAbs(I_x, *(feature[3]), 0.25);
  //     cv::convertScaleAbs(I_y, *(feature[4]), 0.25);

  //     // Orientation of gradients
  //     for(int  y = 0; y < img[0]->rows; y++)
  // 	for(int  x = 0; x < img[0]->cols; x++) {
  // 	  // Avoid division by zero
  // 	  float tx = (float)I_x.at<short>(y, x) + (float)copysign(0.000001f, I_x.at<short>(y, x));
  // 	  // Scaling [-pi/2 pi/2] -> [0 80*pi]
  // 	  feature[1]->at<uchar>(y, x) = (uchar)(( atan((float)I_y.at<short>(y, x) / tx) + 3.14159265f / 2.0f ) * 80);
  // 	  //std::cout << "scaling" << std::endl;
  // 	  feature[2]->at<uchar>(y, x) = (uchar)sqrt((float)I_x.at<short>(y, x)* (float)I_x.at<short>(y, x) + (float)I_y.at<short>(y, x) * (float)I_y.at<short>(y, x));
  // 	}

  //     // Magunitude of gradients
  //     for(int y = 0; y < img[0]->rows; y++)
  // 	for(int x = 0; x < img[0]->cols; x++ ) {
  // 	  feature[2]->at<uchar>(y, x) = (uchar)sqrt(I_x.at<short>(y, x)*I_x.at<short>(y, x) + I_y.at<short>(y, x) * I_y.at<short>(y, x));
  // 	}

  //     hog.extractOBin(feature[1], feature[2], feature, 7);

  //     // calc I_xx I_yy
  //     cv::Sobel(*(feature[0]), I_x, CV_16S, 2, 0);
  //     cv::Sobel(*(feature[0]), I_y, CV_16S, 0, 2);

  //     cv::convertScaleAbs(I_x, *(feature[5]), 0.25);
  //     cv::convertScaleAbs(I_y, *(feature[6]), 0.25);

  //     cv::Mat img_Lab;
  //     cv::cvtColor(*img.at(0), img_Lab, CV_RGB2Lab);
  //     cv::vector<cv::Mat> tempfeature(3);

  //     cv::split(img_Lab, tempfeature);

  //     for(int i = 0; i < 3; ++i)
  // 	tempfeature[i].copyTo(*(feature[i]));

  //     // min max filter
  //     for(int c = 0; c < 16; ++c)
  // 	minFilter(feature[c], feature[c + 16], 5);
  //     for(int c = 0; c < 16; ++c)
  // 	maxFilter(feature[c], feature[c], 5);

  //   }else{
  //      feature.clear();

  // calc gray integral image
  //     cv::Mat grayImg(imgRow, imgCol, CV_8U);
  //     cv::cvtColor(*img[0], grayImg, CV_RGB2GRAY);
  //     integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
  //     cv::integral(grayImg, *integralMat, CV_64F);
  //     feature.push_back(integralMat);

  //     // calc r g b integral image
  //     std::vector<cv::Mat> splittedRgb;
  //     cv::split(*img[0], splittedRgb);
  //     for(unsigned int i = 0; i < splittedRgb.size(); ++i){
  // 	integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
  // 	cv::integral(splittedRgb[i], *integralMat, CV_64F);
  // 	feature.push_back(integralMat);
  //     }

  //     featureFlag = 1;
  //   }
  // }

  // if(img.size() > 1){
  //   cv::Mat tempDepth = cv::Mat(img[0]->rows, img[0]->cols, CV_8U);// = *img.at(1);

  //   if(img[1]->type() != CV_8U)
  //     img[1]->convertTo(tempDepth, CV_8U, 255.0 / (double)(conf.maxdist - conf.mindist));
  //   else
  //     tempDepth = *img[1];
  //   integralMat = new cv::Mat(imgRow + 1, imgCol + 1, CV_64F);
  //   cv::integral(tempDepth, *integralMat, CV_64F);
  //   feature.push_back(integralMat);

  //   featureFlag  = 1;
  // }

  //cv::Mat showDepth = cv::Mat(feature[1]->rows, feature[0]->cols, CV_8U);
  //feature[1]->convertTo(showDepth, CV_8U, 255.0/1000.0);
    
  //    cv::namedWindow("test");
  //    cv::imshow("test", *feature[0]);
  //    cv::namedWindow("test2");
  //    cv::imshow("test2", showDepth);

  //    cv::waitKey(0);

  //    cv::destroyWindow("test");
  //    cv::destroyWindow("test2");


  return 0;
}

int CDataset::releaseFeatures(){
  if(featureFlag == 0){
    std::cout << "image is already released! foolish!" << std::endl;
    return -1;
  }

  for(unsigned int i = 0; i < feature.size(); ++i){
    if(feature[i] != NULL){
      delete feature[i];
      feature[i] = NULL;
    }
  }

  featureFlag = 0;

  return 0;
}

void CDataset::minFilter(cv::Mat* src, cv::Mat* des, int fWind) {
  int d = (fWind - 1) / 2;
  cv::Rect roi;
  cv::Mat desTemp(src->rows, src->cols, CV_8U), vTemp;

  for(int y = 0; y < src->rows - fWind; ++y){ //for image height
    if(y < fWind)
      roi = cv::Rect(0, 0, src->cols, fWind - y);
    else
      roi = cv::Rect(0, y, src->cols, fWind);

    cv::reduce((*src)(roi), vTemp, 0, CV_REDUCE_MIN);

    roi = cv::Rect(0, y + d, src->cols, 1);
    //cv::Mat roiDesTemp(desTemp, roi);
    vTemp.copyTo(desTemp(roi));
  }// For image height

  for(int x = 0; x < src->cols - fWind; ++x){ // for image width
    if(x < d)
      roi = cv::Rect(0, 0, fWind - x, src->rows);
    else
      roi = cv::Rect(x, 0, fWind, src->rows);

    cv::reduce(desTemp(roi), vTemp, 1, CV_REDUCE_MIN);

    roi = cv::Rect(x + d, 0, 1, src->rows);
    cv::Mat roiDesTemp((*des), roi);
    vTemp.copyTo((*des)(roi));// = vTemp.clone();//copyTo((*des)(roi));
  } // for image width
}

void CDataset::maxFilter(cv::Mat* src, cv::Mat* des, int fWind) {
  int d = (fWind - 1) / 2;
  cv::Rect roi;
  cv::Mat desTemp(src->rows, src->cols, CV_8U), vTemp;

  for(int y = 0; y < src->rows - fWind; ++y){ //for image height
    if(y < fWind)
      roi = cv::Rect(0, 0, src->cols, fWind - y);
    else
      roi = cv::Rect(0, y, src->cols, fWind);

    cv::reduce((*src)(roi), vTemp, 0, CV_REDUCE_MAX);

    roi = cv::Rect(0, y + d, src->cols, 1);
    cv::Mat roiDesTemp(desTemp, roi);
    vTemp.copyTo(desTemp(roi));
  }// For image height

  for(int x = 0; x < src->cols - fWind; ++x){ // for image width
    if(x < d)
      roi = cv::Rect(0, 0, fWind - x, src->rows);
    else
      roi = cv::Rect(x, 0, fWind, src->rows);

    cv::reduce(desTemp(roi), vTemp, 1, CV_REDUCE_MAX);

    roi = cv::Rect(x + d, 0, 1, src->rows);
    cv::Mat roiDesTemp(*des, roi);
    vTemp.copyTo((*des)(roi));// = vTemp.clone();//copyTo((*des)(roi));
  } // for image width

}

CNegDataset* convertPosToNeg2(CPosDataset* pos)
{
  CNegDataset* tempNegDataset = new CNegDataset();
  tempNegDataset->setRgbImagePath(pos->getRgbImagePath());
  tempNegDataset->setDepthImagePath(pos->getDepthImagePath());
  //  tempNegDataset->setModelPath(pos->getModelPath());

  tempNegDataset->img.resize(pos->img.size());
  for(unsigned int i = 0; i < pos->img.size(); ++i)
    tempNegDataset->img[i] = pos->img[i];

  tempNegDataset->feature.resize(pos->feature.size());
  for(unsigned int i = 0; i < pos->feature.size(); ++i)
    tempNegDataset->feature[i] = pos->feature[i];

  delete pos;

  return tempNegDataset;
}
