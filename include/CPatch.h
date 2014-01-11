#ifndef __CPATCH__
#define __CPATCH__

//#include "util.h"
#include "CDataset.h"

class CPatch
{
public:
 CPatch(cv::Rect r,CDataset *d) : roi(r), data(d){
        //cv::Mat* depthImage = d->img.at(1);
        //relativePosition.x = data->getDepthImagePath() - roi.x + roi.width / 2 + 1
    }
    CPatch(){}
    virtual ~CPatch(){}

    void setData(CDataset *d){data = d;}
    CDataset* getData()const{return data;}

    void setRoi(cv::Rect r){roi = r;}
    cv::Rect getRoi()const{return roi;}

    cv::Mat* getFeature(int featureNum) const{return data->feature.at(featureNum);}
    cv::Mat* getDepth() const{return data->img.at(1);}
    std::vector<cv::Mat*> getImgs(){return this->data->img;}
    //    int centerPointValue(){return data->img.at(1)->at<ushort>()}
 private:
    cv::Rect roi;
    double scale;
    CDataset *data;
};

class CPosPatch : public CPatch{
public:
  // constructor for setting relative point
  CPosPatch(cv::Rect r,CPosDataset *pos) : CPatch(r, pos),pData(pos){
    relativePosition = pData->getParam()->getCenterPoint() 
      - cv::Point_<double>(getRoi().x + getRoi().width / 2 + 1, 
			   getRoi().y + getRoi().height / 2 + 1);
  }
  //constructor for no setting
  CPosPatch(){}
  virtual ~CPosPatch(){}
    
  // get member and parameters
  std::string getClassName()const{return pData->getParam()->getClassName();}
  cv::Point_<double> getCenterPoint()const{return pData->getParam()->getCenterPoint();}
  int getFeatureNum()const{return pData->feature.size();}
  CParamset getParam()const{return *(pData->getParam());}
  std::string getRgbImageFilePath(){return pData->getRgbImagePath();}

  // set member and parameters
  void setCenterPoint(cv::Point_<double> nCenter){pData->setCenterPoint(nCenter);}
  void setClassName(std::string name){pData->getParam()->setClassName(name);}
  cv::Point_<double> getRelativePosition()const {return relativePosition;}
  void setRelativePosition(const cv::Point_<double> p){relativePosition = p;}

private:
  CPosDataset *pData;
  cv::Point_<double> relativePosition;
};

class CNegPatch : public CPatch{
public:
 CNegPatch(cv::Rect r, CNegDataset *neg ) : CPatch(r, neg), nData(neg){}
    CNegPatch(){}
    int getFeatureNum()const{return nData->feature.size();}
    //cv::Mat* getDepth() const{return nData->img.at(1);}
    virtual ~CNegPatch(){}

private:
    CNegDataset *nData;
};

class CTestPatch : public CPatch{
public:
 CTestPatch(cv::Rect r, CTestDataset *tes) : CPatch(r, tes),tData(tes){}
    CTestPatch(){}
    virtual ~CTestPatch(){}
    //cv::Rect getPatchRoi(){return this->getRoi(
    int getFeatureNum()const {return tData->feature.size();}
    //cv::Mat* getDepth() const{return tData->img.at(1);}

private:
    CTestDataset *tData;
};

#endif
