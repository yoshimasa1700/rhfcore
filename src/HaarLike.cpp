#include "../include/HaarLike.h"

void calcHaarLikeFeature(const cv::Mat &patch, 
                         const int* test, 
                         double &p1, 
                         double &p2){
  double t1,t2,t3,t4;
  double u1,u2,u3,u4;
  
  double rB = (1 - (double)test[2] / 100.0);// ratio of big side
  double rS = (double)test[2] / 100.0;// ratio of small side

  int width = patch.cols * test[3] / 100.0;
  int height = patch.rows * test[4] / 100.0;
  int x = patch.cols * test[5] / 100.0;
  int y = patch.rows * test[6] / 100.0;

  u1 = patch.at<double>(y + height, x + width);//右下
  u2 = patch.at<double>(y, x);//左上
  u3 = patch.at<double>(y + height, x);//左下
  u4 = patch.at<double>(y, x + width);//右上

  double all = u1 + u2 - u3 - u4;
  int menseki = width * height;
  
  p1 = 0;
  p2 = 0;
  int s1 = 0;
  int s2 = 0;

  switch(test[1]){
    case 0:
      {
        //edge feature
        p1 = patch.at<double>(y + height, (x + width) * rS)
            - patch.at<double>(y, (x + width) * rS)
            - patch.at<double>(y + height, x)
            + patch.at<double>(y,x);

        s1 = width * height * rS;
        break;
      }
    case 1:
      // line feature
      p1 = patch.at<double>(y + height,  (1-rS / 2.0) * (x + width))
          - patch.at<double>(y, (x + width) * (1-rS / 2.0))
          - (patch.at<double>(y + height, (x + width) * (rS / 2.0 )))
          + patch.at<double>(y, (x + width) * (rS / 2.0 ));

      s1 = (height) * (width) * rB;
      break;

    case 2:
      {
        // center
        t1 = patch.at<double>((y + height) * rS / 2.0, 
			      (x + width) * rS / 2.0);
        t2 = patch.at<double>((y + height) * rS / 2.0, (x + width) * (1.0 - rS / 2.0));
        t3 = patch.at<double>((y + height) * (1.0 - rS / 2.0), (x + width) * rS / 2.0);
        t4 = patch.at<double>((y + height) * (1.0 - rS / 2.0), (x + width) * (1.0 - rS / 2.0));
        p1 = t4 + t1 - t2 - t3;

        s1 = (height) * (width) * rB * rB;

        break;
      }

    case 3:
      {
        // edge yoko
        p1 = patch.at<double>((y + height) * rS, x + width)
            - patch.at<double>(y, (x + width ))
            - patch.at<double>((y + height) * rS, x)
            + patch.at<double>(y,x);

        s1 = (height) * (width) * rS;
        
        break;
      }
    case 4:
      // line yoko
      p1 = patch.at<double>((1.0 - rS / 2.0) * (y + height),   x + width)
          - patch.at<double>((y + height) * (rS / 2.0), x + width)
          - patch.at<double>((1.0 - rS / 2.0) * (y + height), x)
          + patch.at<double>((y + height) * (rS / 2.0), x);

      s1 = (height) * (width) * rS;

      break;
        
    case 5:
      {
        //edge feature sen taishou
        p1 = all -(patch.at<double>(y + height, (x + width) * rS)
          - patch.at<double>(y, (x + width) * rS)
          - patch.at<double>(y + height, x)
          + patch.at<double>(y,x));
        // - (patch.at<double>(y + height, (x + width) * (double)test[2] / 100.0)
        //             - patch.at<double>(y, (x + width ) * (double)test[2] / 100.0)
        //             - patch.at<double>(y + height, x)
        //             + patch.at<double>(y,x));
        s1 = menseki - (height) * (width) * rS;

        break;
      }
    case 6:
      {
        // edge yoko sen taishou
        p1 = all - (patch.at<double>(y + height, (x + width) * rS)
                    - patch.at<double>(y, (x + width) * rS)
                    - patch.at<double>(y + height, x)
                    + patch.at<double>(y,x));

        // (patch.at<double>((y + height) * (double)test[2] / 100.0, x + width)
        //             - patch.at<double>(y, (x + width ))
        //             - patch.at<double>((y + height) * (double)test[2] / 100.0, x)
        //             + patch.at<double>(y,x));
        // t1 = patch.at<double>(y + height, x + width)
        //     - (patch.at<double>(y + height, 0)
        //        + patch.at<double>(0, x + width))
        //     + patch.at<double>(0,0);

        // p2 = t1 - p1;


        s1 = menseki - (height) * (width) * (double)test[2] / 100.0;
        //        s2 = (height) * (width) * (1.0 - (double)test[2] / 100.0);
        // double temp;
        // temp = s2;
        // s2 = s1;
        // s1 = temp;
        
        // temp = p2;
        // p2 = p1;
        // p1 = temp;
        break;
      }
    default:
      break;
  }

  p2 = all - p1;
  s2 = menseki - s1;

  //  for(int i = 0; i < 8; ++i)
  //     std::cout << test[i] << " ";
  //   std::cout << std::endl;

  // std::cout << "p1 " << p1 << " p2 " << p2 << std::endl;
  // std::cout << "s1 " << s1 << " s2 " << s2 << std::endl;

  if(s1 == 0 || s2 == 0){
    std::cout << "error! can't get enough space for haar-like feature" << std::endl;
    std::cout << s1 << " " << s2 << std::endl;

    for(int i = 0; i < 8; ++i)
      std::cout << test[i] << " ";
    std::cout << std::endl;

    exit(-1);
  }else{
    p1 /= (double)s1;
    p2 /= (double)s2;
  }

  return;
}
