/* 
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include <vector>
#include <iostream>
#include "HoG.h"

using namespace std;

HoG::HoG() {
    bins = 9;
    binsize = (3.14159265f*80.0f)/float(bins);;

    g_w = 5;
    Gauss = cvCreateMat( g_w, g_w, CV_32FC1 );
    double a = -(g_w-1)/2.0;
    double sigma2 = 2*(0.5*g_w)*(0.5*g_w);
    double count = 0;
    for(int x = 0; x<g_w; ++x) {
        for(int y = 0; y<g_w; ++y) {
            double tmp = exp(-( (a+x)*(a+x)+(a+y)*(a+y) )/sigma2);
            count += tmp;
            cvSet2D( Gauss, x, y, cvScalar(tmp) );
        }
    }
    cvConvertScale( Gauss, Gauss, 1.0/count);

    ptGauss = new float[g_w*g_w];
    int i = 0;
    for(int y = 0; y<g_w; ++y)
        for(int x = 0; x<g_w; ++x)
            ptGauss[i++] = (float)cvmGet( Gauss, x, y );

    currentF = 0;
}


void HoG::extractOBin(const cv::Mat* Iorient,const cv::Mat* Imagn, std::vector<cv::Mat*>& out, int off) {
  //    double* desc = new double[bins];

    for(int k=off; k<bins+off; ++k){
        int r = out[k + currentF]->rows;
        int c = out[k + currentF]->cols;
        *(out[k]) = cv::Mat::zeros(r,c, CV_8U);
    }

    for(int y = 0; y < Iorient->rows - g_w; y++) {
        for(int x = 0; x < Iorient->cols - g_w; x++){
            // calc hog bin
            calcHoGBin(Iorient, Imagn, out, x, y);
        }
    }

//    std::cout << "output hog image" << std::endl;
//    for(int i = off; i < bins + off; ++i){
//        cv::namedWindow("test");
//        cv::imshow("test",*out.at(i));
//        cv::waitKey(0);
//        cv::destroyWindow("test");
//        std::stringstream ss;
//        ss << i;
//        std::string filename = "HOG" + ss.str() + ".png";
//        cv::imwrite(filename,*out.at(i));
//    }
//    std::cout << "hog image output end" << std::endl;
}

void HoG::calcHoGBin(const cv::Mat* IOri, const cv::Mat* IMag, std::vector<cv::Mat*>& out, int offX, int offY){
    int yy, xx;

    int i = 0;
    for(int y = 0; y < g_w; ++y){
        for(int x = 0; x < g_w; ++x, ++i){
            yy = y + offY;
            xx = x + offX;

            float v = (float)IOri->at<uchar>(yy, xx) / binsize;
            float w = (float)IMag->at<uchar>(yy, xx) * ptGauss[i];
            int bin1 = int(v);
            int bin2;
            float delta = v - bin1 - 0.5f;
            if(delta < 0){
                bin2 = bin1 < 1 ? bins - 1 : bin1 - 1;
                delta = - delta;
            }else
                bin2 = bin1 < bins - 1 ? bin1 + 1 : 0;
            out.at(bin1 + 7 + currentF)->at<uchar>(yy, xx) += (1 - delta) * w;
            out.at(bin2 + 7 + currentF)->at<uchar>(yy, xx) += delta * w;
        }
    }
}


