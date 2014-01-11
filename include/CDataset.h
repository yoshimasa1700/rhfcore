#ifndef CDATASET_H
#define CDATASET_H


#include <iostream>
#include <sstream>

#include "CConfig.h"
#include "HoG.h"

class CParamset{
public:
    CParamset(){}
    ~CParamset(){}

    int setCenterPoint(cv::Point cp){centerPoint = cp;return 0;}
    cv::Point_<double> getCenterPoint()const{return centerPoint;}

    int setRelativePoint(cv::Point cp){relativePoint = cp;return 0;}
    cv::Point_<double> getRelativePoint()const{return relativePoint;}

    int setClassName(std::string name){className = name;return 0;}
    std::string getClassName()const{return className;}

    int setAngle(double* an){
        for(int i =0; i < 3; ++i)
            angle[i] = an[i];
        return 0;
    }
    const double* getAngle()const{return angle;}

    int showParam();

    CParamset& operator+=(const CParamset& obj);
    CParamset& operator/=(const float& div);

    std::string outputParam();
    //void readParam(std::stringstream *in);

private:
    // parameters should be estimated
    cv::Point_<double> centerPoint;
    cv::Point_<double> relativePoint;
    std::string className;
    double angle[3];

    cv::Rect boundingbox;
};

// !!caution!!
// this is abstract class you should inherit this class!
class CDataset {
public:
    CDataset();
    virtual ~CDataset();

    int loadImage(const CConfig &);
    //int loadImage(CGlObjLoader *obj,const CConfig &conf, const std::string modelName, const CParamset *param);
    int releaseImage();

    int extractFeatures(const CConfig &);
    int releaseFeatures();

    void setRgbImagePath(std::string rgb_path){rgb = rgb_path;}
    void setDepthImagePath(std::string depth_path){depth = depth_path;}
    void setMaskImagePath(std::string mask_path){mask = mask_path;}

    std::string getRgbImagePath(){return rgb;}
    std::string getDepthImagePath(){return depth;}
    std::string getMaskImagePath(){return mask;}

    /* void setModelPath(std::string p){model = p;} */
    /* std::string getModelPath(){return model;} */

    int calcHoG(int type);

    // loaded images and features
    std::vector<cv::Mat*> img, feature;

    //CGlObjLoader obj;

private:
    // flag for image or features loaded on memory
    bool imgFlag, featureFlag;

    // image file path
    std::string rgb, depth, mask;
    std::string model;

    // min and max filter
    void minFilter(cv::Mat* src, cv::Mat* des, int fWind);
    void maxFilter(cv::Mat* src, cv::Mat* des, int fWind);

    //  cv::Rect bBox;
    //  std::vector<std::string> className;
    //  std::vector<cv::Poinkt> centerPoint;
    //  std::vector<double> angles;

    //void showDataset();

    HoG hog;
};

class CPosDataset : public CDataset{
public:
    CPosDataset(){}
    virtual ~CPosDataset(){}

    void setClassName(std::string name){param.setClassName(name);}
    void setAngle(double* an){param.setAngle(an);}
    void setCenterPoint(cv::Point_<double> cp){param.setCenterPoint(cp);}

    std::string getClassName(){return param.getClassName();}
    CParamset* getParam(){return &param;}
    void setParam(CParamset p){param = p;}

private:
    CParamset param;
};

class CNegDataset : public CDataset{
public:
    CNegDataset(){}
    virtual ~CNegDataset(){}

};

class CTestDataset : public CDataset{
public:
    CTestDataset(){}
    virtual ~CTestDataset(){}

    std::vector<CParamset> param;
};


void cropImageAndDepth(cv::Mat* rgb, cv::Mat* depth, double mindist, double maxdist);


CNegDataset* convertPosToNeg2(CPosDataset* pos);

#endif // CDATASET_H
