#include "../include/util.h"

boost::lagged_fibonacci1279 nCk::gen = boost::lagged_fibonacci1279();
boost::lagged_fibonacci1279 genPose = boost::lagged_fibonacci1279();

float calcSumOfDepth(cv::Mat &depth, const CConfig &conf){
  cv::Mat convertedDepth = cv::Mat(depth.rows, depth.cols, CV_8U);
  cv::Mat integralMat = cv::Mat(depth.rows + 1, depth.cols+1, CV_32F);
  cv::Mat temp1,temp2;
  depth.convertTo(convertedDepth,CV_8U,255.0/(double)(conf.maxdist - conf.mindist));

  cv::integral(convertedDepth,integralMat,temp1,temp2,CV_32F);
  return integralMat.at<int>(depth.rows, depth.cols);
}

// void loadTrainObjFile(CConfig conf, std::vector<CPosDataset*> &posSet)
// {
//   std::vector<std::string> modelPath(0);
//   std::vector<std::string> modelName(0);
//   std::string trainModelListPath = conf.modelListFolder + PATH_SEP + conf.modelListName;

//   boost::uniform_real<> dst(0, 360);
//   boost::variate_generator<boost::lagged_fibonacci1279&,
// 			   boost::uniform_real<> > rand(genPose, dst);

//   posSet.clear();

//   std::ifstream modelList(trainModelListPath.c_str());
//   if(!modelList.is_open()){
//     std::cout << "train model list is not found!" << std::endl;
//     exit(-1);
//   }

//   int modelNum = 0;
//   modelList >> modelNum;
//   modelPath.resize(modelNum);
//   modelName.resize(modelNum);

//   for(int i = 0; i < modelNum; ++i){
//     std::string tempName;
//     modelList >> tempName;
//     modelPath[i] = conf.modelListFolder +PATH_SEP + tempName;
//     std::string tempClass;
//     modelList >> tempClass;
//     modelName[i] = tempClass;
//   }

//   posSet.resize(modelNum * conf.imagePerTree);
    
//   //    std::cout << modelNum << std::endl;
//   for(int j = 0; j < modelNum; ++j){
//     for(int i = 0; i < conf.imagePerTree; ++i){
//       CPosDataset* posTemp = new CPosDataset();
//       posTemp->setModelPath(modelPath.at(j));
//       posTemp->setClassName(modelName.at(j));
//       double tempAngle[3];
//       for(int k = 0; k < 3; ++k)
// 	tempAngle[k] = rand();
//       posTemp->setAngle(tempAngle);
//       posTemp->setCenterPoint(cv::Point(320,240));

//       posSet[j * conf.imagePerTree + i] = posTemp;
//     }
//   }
//   modelList.close();
// }

void loadTrainPosFile(CConfig conf, std::vector<CPosDataset*> &posSet)
{
  std::string traindatafilepath = conf.trainpath + PATH_SEP +  conf.traindatafile;
  int n_folders;
  int n_files;
  std::vector<std::string> trainimagefolder;
  std::vector<CPosDataset*> tempDataSet;
  std::string trainDataListPath;
  //int dataSetNum = 0;
  CClassDatabase database;
  cv::Point tempPoint;
  nCk nck;

  posSet.clear();

  //read train data folder list
  std::ifstream in(traindatafilepath.c_str());
  if(!in.is_open()){
    std::cout << "train data floder list " << traindatafilepath.c_str() << " is not found!" << std::endl;
    exit(1);
  }

  // read train pos folder
  in >> n_folders;
  std::cout << "number of training data folders : " << n_folders << std::endl;
  trainimagefolder.resize(n_folders);
  for(int i = 0;i < n_folders; ++i)
    in >> trainimagefolder.at(i);
  //  std::cout << trainimagefolder.at(0) << std::endl;

  // close train pos data folder list
  in.close();

  std::cout << "train folders lists : " << std::endl;
  //read train file name and grand truth from file
  tempDataSet.clear();
  for(int i = 0;i < n_folders; ++i){
    trainDataListPath
      = conf.trainpath + PATH_SEP + trainimagefolder.at(i) + PATH_SEP + conf.traindatalist;

    std::cout << trainDataListPath << std::endl;
    std::string imageFilePath
      = conf.trainpath + PATH_SEP + trainimagefolder.at(i) + PATH_SEP;

    std::ifstream trainDataList(trainDataListPath.c_str());
    if(!trainDataList.is_open()){
      std::cout << "can't read " << trainDataListPath << std::endl;
      exit(1);
    }

    trainDataList >> n_files;

    // if invarid data in image list, error
    try{
      for(int j = 0;j < n_files; ++j){
	CPosDataset* posTemp = new CPosDataset();
	std::string nameTemp;

	//read file names
	trainDataList >> nameTemp;
	posTemp->setRgbImagePath(imageFilePath + nameTemp);
	if(nameTemp.empty())
	  throw nameTemp;
	if(conf.learningMode != 1){
	  cv::Mat tm = cv::imread(posTemp->getRgbImagePath());
	  if(tm.data == NULL)
	    throw posTemp->getRgbImagePath();
	}

	trainDataList >> nameTemp;
	posTemp->setDepthImagePath(imageFilePath + nameTemp);
	if(nameTemp.empty())
	  throw nameTemp;
	if(conf.learningMode != 2){
	  cv::Mat tm = cv::imread(posTemp->getDepthImagePath(), CV_LOAD_IMAGE_ANYDEPTH);
	  if(tm.data == NULL)
	    throw posTemp->getDepthImagePath();
	}

	trainDataList >> nameTemp;// dummy
	posTemp->setMaskImagePath(imageFilePath + nameTemp);
	if(nameTemp.empty())
	  throw 0;
	// if(conf.learningMode != 1){
	//   cv::Mat tm = cv::imread(posTemp->getMaskImagePath());
	//   if(tm.data == NULL)
	//     throw posTemp->getMaskImagePath();
	// }

	//read class name
	std::string tempClassName;
	trainDataList >> tempClassName;
	posTemp->setClassName(tempClassName);
	if(nameTemp.empty())
	  throw 0;

	// read image size
	cv::Mat tempImage = cv::imread(posTemp->getRgbImagePath(),3);
	cv::Size tempSize = cv::Size(tempImage.cols, tempImage.rows);

	// set center point
	tempPoint = cv::Point(tempImage.cols / 2, tempImage.rows / 2);
	posTemp->setCenterPoint(tempPoint);

	// registre class param to class database
	database.add(posTemp->getParam()->getClassName(), tempSize, 0);

	//read angle grand truth
	double tempAngle[3];
	for(int i = 0; i < 3; ++i)
	  trainDataList >> tempAngle[i];
	posTemp->setAngle(tempAngle);

	tempDataSet.push_back(posTemp);
      }
    } // try

    catch (std::string e){
      std::cout << "can't found image in pos image list, image name is " << e  << std::endl;
      exit(-1);
    }
    catch (int i){
      std::cout << "null data in image list please check pos image num" << std::endl;
      exit(-1);
    }
    catch (...){
      std::cout << "some undefined error occurred while reading pos image list" << std::endl;
      exit(-1);
    }

    trainDataList.close();
  }

  int dataOffset = 0;
  unsigned int classNum =  database.vNode.size();
  for(unsigned int j = 0;j < classNum; j++){
    std::set<int> chosenData = nck.generate(database.vNode[j].instances, conf.imagePerTree);
    std::set<int>::iterator ite = chosenData.begin();

    while(ite != chosenData.end()){
      posSet.push_back(tempDataSet.at(*ite + dataOffset));
      ite++;
    }
    dataOffset += database.vNode.at(j).instances;
  }
}

void loadTrainNegFile(CConfig conf, std::vector<CNegDataset*> &negSet)
{
  //std::string traindatafilepath
  int n_folders;
  std::ifstream in_F((conf.negDataPath + PATH_SEP + conf.negFolderList).c_str());
  in_F >> n_folders;
  std::cout << conf.negDataPath + PATH_SEP + conf.negFolderList << std::endl;


  for(int j = 0; j < n_folders; ++j){
    int n_files;
    CDataset temp;
    //std::string trainDataListPath = conf.negDataPath + PATH_SEP +  conf.negDataList;

    std::string negDataFolderName;
    in_F >> negDataFolderName;
    std::string negDataFilePath = conf.negDataPath + PATH_SEP + negDataFolderName +  PATH_SEP;

    //read train data folder list
    std::ifstream in((negDataFilePath + conf.negDataList).c_str());
    if(!in.is_open()){
      std::cout << negDataFilePath << " train negative data floder list is not found!" << std::endl;
      exit(1);
    }

    std::cout << negDataFilePath << " loaded!" << std::endl;
    in >> n_files;

    try{
      for(int i = 0; i < n_files; ++i){
	CNegDataset* negTemp = new CNegDataset();
	std::string tempName;

	in >> tempName;
	negTemp->setRgbImagePath(negDataFilePath + tempName);
	if(conf.learningMode != 1){
	  //	  std::cout << negTemp->getRgbImagePath() << std::endl;
	  cv::Mat tm = cv::imread(negTemp->getRgbImagePath());
	  if(tm.data == NULL)
	    throw negTemp->getRgbImagePath();
	}	
	if(tempName.empty())
	  throw tempName;


	in >> tempName;
	negTemp->setDepthImagePath(negDataFilePath + tempName);
	if(conf.learningMode != 2){
	  //	  std::cout << negTemp->getDepthImagePath() << std::endl;
	  cv::Mat tm = cv::imread(negTemp->getDepthImagePath(), CV_LOAD_IMAGE_ANYDEPTH);
	  if(tm.data == NULL)
	    throw negTemp->getDepthImagePath();
	}
	if(tempName.empty())
	  throw 0;

	negSet.push_back(negTemp);
      }
    }
    catch (std::string e){
      std::cout << "can't found image in neg image list, image name is " << e << std::endl;
      exit(-1);
    }
    catch (int i){
      std::cout << "null data in neg image list please check image" << std::endl;
      exit(-1);
    }
    catch (...){
      std::cout << "some undefined error occurred while reading neg image list" << std::endl;
      exit(-1);
    }

    in.close();
  }
  in_F.close();
}

// extract patch from images
// !!!!!!coution!!!!!!!
// this function is use for negatime mode!!!!!
void extractPosPatches(std::vector<CPosDataset*> &posSet,
                       std::vector<CPosPatch> &posPatch,
                       CConfig conf,
                       const int treeNum,
                       const CClassDatabase &classDatabase){
  cv::Rect tempRect;
  std::vector<CPosPatch> tPosPatch(0);//, posPatch(0);
  std::vector<std::vector<CPosPatch> > patchPerClass(classDatabase.vNode.size());
  //int pixNum;
  nCk nck;
  int classNum = 0;
  cv::Mat roi;

  // cv::Mat showDepth = cv::Mat(posSet[0]->img[1]->rows, posSet[0]->img[1]->cols, CV_8U);
  // posSet[0]->img[1]->convertTo(showDepth, CV_8U, 255.0/1000.0);
    
  // cv::namedWindow("test");
  //    cv::imshow("test", *posSet[0]->img[0]);
  //    cv::namedWindow("test2");
  //    cv::imshow("test2", showDepth);

  //    cv::waitKey(0);

  //    cv::destroyWindow("test");
  //    cv::destroyWindow("test2");


  //    std::cout << posSet[1]->img[1]->type() << " " << CV_16U << std::endl;

  posPatch.clear();

  tempRect.width  = conf.p_width;
  tempRect.height = conf.p_height;
  

  std::cout << "image num is " << posSet.size() << std::endl;;

  std::cout << "extracting pos patch from image" << std::endl;
  for(unsigned int l = 0;l < posSet.size(); ++l){

    for(int j = conf.p_width; j < posSet.at(l)->img.at(0)->cols - conf.p_width; j += conf.stride){
      for(int k = conf.p_height; k < posSet.at(l)->img.at(0)->rows - conf.p_height; k += conf.stride){
	tempRect.x = j;
	tempRect.y = k;

	// set patch class
	classNum = classDatabase.search(posSet.at(l)->getParam()->getClassName());//dataSet.at(l).className.at(0));
	if(classNum == -1){
	  std::cout << "class not found!" << std::endl;
	  exit(-1);
	}

	//tPatch.setPatch(temp, image.at(l), dataSet.at(l), classNum);

	CPosPatch posTemp(tempRect, posSet.at(l));

        // std::cout << tempRect << std::endl;
        // std::cout << posTemp.getRelativePosition() << std::endl;
	int centerDepthFlag = 1;

	if(conf.learningMode != 2){
	  cv::Mat tempDepth1 = *posTemp.getDepth();
	  cv::Mat tempDepth2 = tempDepth1(tempRect);

	  // if()
	  //   centerDepthFlag = 1;
	  //std::cout << tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) << std::endl;
	
	// cv::namedWindow("test2");
          // depth->convertTo(showDepth, CV_8U, 255.0/1000);
          // cv::imshow("test2", showDepth);
          // cv::waitKey(0);

          // cv::Mat showDepth;
          // cv::namedWindow("test2");
          // tempDepth2.convertTo(showDepth, CV_8U, 255.0/1000);
          // cv::imshow("test2", showDepth);
          // cv::waitKey(0);
          // std::cout << tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) << std::endl;
          
	  //	std::cout << centerDepthFlag << std::endl;

	  //if (conf.learningMode == 2){// || pixNum > 0){

	  //std::cout << tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) << std::endl;
	  if(tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) != 0
             && tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1 < 1000)){
	    //	  std::cout << "test" << std::endl;
	    //if(conf.learningMode != 2){
	    //	    std::cout << (double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) << std::endl;
            normarizationByDepth(posTemp , conf, (double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1));
	    normarizationCenterPointP(posTemp, conf,(double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1));
	    //}

	    //	    std::cout << posTemp.getRoi() << std::endl;
	    
	  
	    //	  std::cout << "kokomade" << std::endl;
	    //std::cout << posTemp.getRoi().width << std::endl;
	  }else{
	    centerDepthFlag = 0;
            //            std::cout << "zero deshi ta" << std::endl;
          }
	}
	if(/*posTemp.getRoi().width > 5 && posTemp.getRoi().height > 5 &&*/ centerDepthFlag){
	  std::vector<double> pRatio(0);
          //          pRatio.push_back(0.5);
	  pRatio.push_back(1.0);
	  //pRatio.push_back(1.4);
          //        	  pRatio.push_back(1.5);
          // if(conf.learningMode == 2){
          //   pRatio.push_back(2.0);
          //   //            pRatio.push_back(2.5);
          //   pRatio.push_back(3.0);
          // }
	  // pRatio.push_back(3.4);
	  // //	  pRatio.push_back(3.8);
	  // pRatio.push_back(4.2);
	  // //	  pRatio.push_back(4.6);
	  // pRatio.push_back(5.0);

	  // pRatio.push_back(1.0);
	  // // pRatio.push_back(1.2);
	  // pRatio.push_back(1.4);
	  // // pRatio.push_back(1.6);
	  // pRatio.push_back(1.8);
	  // //	  pRatio.push_back(1.5);
	  //   pRatio.push_back(2.0);

	  for(unsigned int r = 0; r < pRatio.size(); ++r){
	    CPosPatch transPatch = posTemp;  // copy constructor
	    cv::Rect tempRoi = transPatch.getRoi();
	    
	    if(tempRoi.x - (int)((double)tempRoi.width * (pRatio[r] - 1.0) / 2.0)> 0)
	      tempRoi.x -= (int)((double)tempRoi.width * (pRatio[r] - 1.0) / 2.0);
	    else
	      tempRoi.x = 0;

	    if(tempRoi.y - (int)((double)tempRoi.height * (pRatio[r] - 1.0) / 2.0)> 0)
	      tempRoi.y -= (int)((double)tempRoi.height * (pRatio[r] - 1.0) / 2.0);
	    else
	      tempRoi.y = 0;

	    if(tempRoi.width * pRatio[r] + tempRoi.x < posSet.at(l)->img.at(0)->cols)
	      tempRoi.width *= pRatio[r];
	    else
	      tempRoi.width = posSet.at(l)->img.at(0)->cols - tempRoi.x;

	    if(tempRoi.height * pRatio[r] + tempRoi.y < posSet.at(l)->img.at(0)->rows)
	      tempRoi.height *= pRatio[r];
	    else
	      tempRoi.height = posSet.at(l)->img.at(0)->rows - tempRoi.y;

	    transPatch.setRoi(tempRoi);
            //            std::cout << "haitte masu" << std::endl;
	    cv::Point_<double> tempRP = transPatch.getRelativePosition();
            //            std::cout << tempRP << std::endl;
	    tempRP.x /= pRatio[r];
	    tempRP.y /= pRatio[r];

	    transPatch.setRelativePosition(tempRP);



	    // tPosPatch.push_back(posTemp);
	    // patchPerClass.at(classNum).push_back(posTemp);
	   
	    int totalZeroPoint = 0;
	    int totalPoint = 0;
	    if(conf.learningMode != 2){

	      for(int m = 0; m < tempRoi.width; ++m){
		for(int n = 0; n < tempRoi.height; ++n){
		  if((*transPatch.getDepth())(tempRoi).at<ushort>(n,m) == 0)
		    totalZeroPoint += 1;
		  totalPoint += 1;
		}
	      }
	    }

	    //	    std::cout << totalZeroPoint << " " << totalPoint << std::endl;

	    if( (double)totalZeroPoint < (double)totalPoint * 0.3 || conf.learningMode == 2){
	      tPosPatch.push_back(transPatch);
	      patchPerClass.at(classNum).push_back(transPatch);
	    }
	  }
	}

	//} // if
      }//x
    }//y
  }//allimages

  // for(unsigned int w = 0; w < patchPerClass.size(); ++w){
  //   std::cout << patchPerClass.at(w).size() << " ";
  // }
  // std::cout << std::endl;

  std::vector<int> patchNum(patchPerClass.size(),conf.patchRatio);

  for(unsigned int i = 0; i < patchPerClass.size(); ++i){
    if(i == treeNum % patchPerClass.size())
      patchNum.at(i) = conf.patchRatio;
    else
      patchNum.at(i) = conf.patchRatio * conf.acPatchRatio;
  }

  // for(unsigned int w = 0; w < patchPerClass.size(); ++w){
  //   std::cout << patchPerClass.at(w).size() << " ";
  // }

  // choose patch from each image
  for(unsigned int i = 0; i < patchPerClass.size(); ++i){
    if(patchPerClass.at(i).size() > conf.patchRatio){

      std::set<int> chosenPatch = nck.generate(patchPerClass.at(i).size(),patchNum.at(i));// conf.patchRatio);//totalPatchNum * conf.patchRatio);
      std::set<int>::iterator ite = chosenPatch.begin();

      while(ite != chosenPatch.end()){
	//std::cout << "this is for debug ite is " << tPosPatch.at(*ite).center << std::endl;
	//std::cout <<posPatch.at(i)
	//std::cout << patchPerClass.at(i).at(*ite).getRgbImageFilePath() << std::endl;
	posPatch.push_back(patchPerClass.at(i).at(*ite));
	ite++;
      }
    }//else{
      //                std::cout << classNum << std::endl;
      ////            cv::namedWindow("test");
      ////            cv::imshow("test", *posSet.at(0).img.at(0));
      ////            cv::waitKey(0);
      ////            cv::destroyWindow("test");
      //            //std::cout << *posSet.at(1).img.at(1) << std::endl;
      //                std::cout << patchPerClass.size() << std::endl;

      //std::cout << "can't extruct enough patch" << std::endl;
      //exit(-1);
    //    }
  }
  return;
}

void extractNegPatches(std::vector<CNegDataset*> &negSet,
                       std::vector<CNegPatch> &negPatch,
                       CConfig conf){
  cv::Rect tempRect;

  std::vector<CNegPatch> tNegPatch(0);//, posPatch(0);
  nCk nck;
  unsigned int negPatchNum = 0;

  negPatch.clear();

  tempRect.width  = conf.p_width;
  tempRect.height = conf.p_height;

  // extract negative patch
  for(unsigned int i = 0; i < negSet.size(); ++i){
    for(int j = 0; j < negSet.at(i)->img.at(0)->cols - conf.p_width; j += conf.stride){
      for(int k = 0; k < negSet.at(i)->img.at(0)->rows - conf.p_height; k += conf.stride){

	tempRect.x = j;
	tempRect.y = k;

	CNegPatch negTemp(tempRect, negSet.at(i));

	int centerDepthFlag = 0;

	if(conf.learningMode != 2){
	  cv::Mat tempDepth1 = *negTemp.getDepth();
	  cv::Mat tempDepth2 = tempDepth1(tempRect);

	  if(tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) != 0){

	    normarizationByDepth(negTemp ,
                                 conf,
                                 (double)tempDepth2.at<ushort>(conf.p_height / 2 + 1,
                                                               conf.p_width / 2 + 1));


	  }
	}

	if(negTemp.getRoi().width > 5 && negTemp.getRoi().height > 5)
	  tNegPatch.push_back(negTemp);
	//}
      }//x
    }//y
  } // every image

    // choose negative patch randamly
  negPatchNum = conf.patchRatio * conf.pnRatio;
  //std::cout << "pos patch num : " << posPatch.size() << " neg patch num : " << negPatchNum << std::endl;

  if(negPatchNum < tNegPatch.size()){
    std::set<int> chosenPatch = nck.generate(tNegPatch.size(), negPatchNum);//totalPatchNum * conf.patchRatio);
    std::set<int>::iterator ite = chosenPatch.begin();

    while(ite != chosenPatch.end()){
      negPatch.push_back(tNegPatch.at(*ite));
      ite++;
    }
  }else{
    std::cout << "only " << tNegPatch.size() << " pathes extracted from negative images" << std::endl;
    std::cout << "can't extract enogh negative patch please set pnratio more low!" << std::endl;
    exit(-1);
  }

  std::cout << "neg patch extraction end" << std::endl;

  //    patches.push_back(posPatch);
  //    patches.push_back(negPatch);
}

void extractTestPatches(CTestDataset* testSet,std::vector<CTestPatch> &testPatch, CConfig conf){

  cv::Rect tempRect;

  tempRect.width = conf.p_width;
  tempRect.height = conf.p_height;

  int imgCol = testSet->img[0]->cols;
  int imgRow = testSet->img[0]->rows;

  //  std::cout << imgCol << " " << imgRow << std::endl;
  //  std::cout << conf.learningMode << std::endl;
  
  testPatch.clear();
  testPatch.reserve((int)(((double)imgCol / (double)conf.stride) * ((double)imgRow / (double)conf.stride)));

  for(int j = 0; j < imgCol - conf.p_width; j += conf.stride){
    for(int k = 0; k < imgRow - conf.p_height; k += conf.stride){
      tempRect.x = j;
      tempRect.y = k;

      //      std::cout << tempRect << std::endl;

      CTestPatch testTemp(tempRect, testSet);
      int centerDepthFlag = 1;

      if(conf.learningMode != 2){
	cv::Mat tempDepth1 = *testTemp.getDepth();
	cv::Mat tempDepth2 = tempDepth1(tempRect);

	if(tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) == 0 ||
	   tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1) == 1024){
	  centerDepthFlag = 0;
	  
	}else
	  normarizationByDepth(testTemp , 
			       conf, 
			       (double)tempDepth2.at<ushort>(conf.p_height / 2 + 1, conf.p_width / 2 + 1));

      }

      int totalZeroPoint = 0;
      int totalPoint = 0;
      if(conf.learningMode != 2){

        for(int m = 0; m < tempRect.width; ++m){
          for(int n = 0; n < tempRect.height; ++n){
            if((*testTemp.getDepth())(tempRect).at<ushort>(n,m) == 0)
              totalZeroPoint += 1;
            totalPoint += 1;
          }
        }
      }else
        totalZeroPoint = totalPoint;


      if(testTemp.getRoi().width > 5 && testTemp.getRoi().height > 5 && centerDepthFlag == 1){
        if(conf.learningMode == 2 || (double)totalZeroPoint < (double)totalPoint * 0.3){
          testPatch.push_back(testTemp);
          //          std::cout << testTemp.getRoi() << std::endl;
        }
      }
      //}
    }
  }
}

void pBar(int p,int maxNum, int width){
  int i;
  std::cout << "[";// << std::flush;
  for(i = 0;i < (int)((double)(p + 1)/(double)maxNum*(double)width);++i)
    std::cout << "*";

  for(int j = 0;j < width - i;++j)
    std::cout << " ";

  std::cout << "]" << (int)((double)(p + 1)/(double)maxNum*100) << "%"  << "\r" << std::flush;
}

void CClassDatabase::add(std::string str, cv::Size size, uchar depth){
  for(unsigned int i = 0; i < vNode.size(); ++i){
    if(str == vNode.at(i).name){
      vNode.at(i).instances++;
      return;
    }
  }
  //std::cout << str << " " << size << " " << depth << std::endl;
  vNode.push_back(databaseNode(str,size,depth));
  return;
}

void CClassDatabase::write(const char* str){

  std::ofstream out(str);
  if(!out.is_open()){
    std::cout << "can't open " << str << std::endl;
    return;
  }

  for(unsigned int i = 0; i < vNode.size(); ++i){
    out << i << " " << vNode.at(i).name
	<< " " << vNode.at(i).classSize.width << " " << vNode.at(i).classSize.height
	<< " " << vNode.at(i).classDepth
	<< std::endl;
  }
  out.close();

  //  std::cout << "out ha shimemashita" << std::endl;
}

int CClassDatabase::read(const char* str){
  std::string tempStr;
  std::stringstream tempStream;
  int tempClassNum;
  std::string tempClassName;
  cv::Size tempSize;
  uchar tempDepth;


  std::ifstream in(str);
  if(!in.is_open()){
    std::cout << "can't open " << str << std::endl;
    return -1;
  }

  //std::cout << str << std::endl;

  //vNode.clear();

  do{
    in >> tempClassNum;
    in >> tempClassName;
    in >> tempSize.width;
    in >> tempSize.height;
    in >> tempDepth;
    in.ignore();
    if(!in.eof())
      this->add(tempClassName,tempSize,tempDepth);

  }while(!in.eof());

  in.close();
  return 0;
}

void CClassDatabase::show() const{
  if(vNode.size() == 0){
    std::cout << "No class registerd" << std::endl;
    return;
  }

  for(unsigned int i = 0; i < vNode.size(); ++i){
    std::cout << "class:" << i << " name:" << vNode.at(i).name << " has " << vNode.at(i).instances << " instances" << std::endl;
  }
}

int CClassDatabase::search(std::string str) const{
  for(unsigned int i = 0; i < vNode.size(); i++){
    //std::cout << i << " " << str << " " << vNode.at(i).name << std::endl;
    if(str == vNode.at(i).name)return i;
  }
  return -1;
}

int normarizationByDepth(CPatch &patch, const CConfig &config, double cd){//, const CConfig &config)const {

  // if(cd == 0){
  //   std::cerr << "error! depth is 0, something wrong" << std::endl;
  //   exit(-1);
  // }


  // cv::Mat tempFeature = *patch.getFeature(4);
  // cv::Rect tempRect = patch.getRoi();
  // cv::Mat realFeature = tempFeature(patch.getRoi());

  // double a = realFeature.at<double>(0,0) + realFeature.at<double>(tempRect.height,tempRect.width) - realFeature.at<double>(0,tempRect.width) - realFeature.at<double>(tempRect.height, 0);
  // a /= tempRect.height;
  // a /= tempRect.width;
  
  // cv::Rect roi;
  // double sca = 1 - (500.0 - a) / 500.0;

  // roi.width = patch.getRoi().width / sca;
  // roi.height = patch .getRoi().height / sca;

  // roi.x = patch.getRoi().x - roi.width / 2;
  // roi.y = patch.getRoi().y - roi.height / 2;

  // if(roi.x < 0) roi.x = 0;
  // if(roi.y < 0) roi.y = 0;
  // if(roi.x + roi.width > patch.getDepth()->cols) roi.width = patch.getDepth()->cols - roi.x;
  // if(roi.y + roi.height > patch.getDepth()->rows) roi.height = patch.getDepth()->rows - roi.y;

  // patch.setRoi(roi);

  // return 0;
}

int normarizationCenterPointP(CPosPatch &patch, const CConfig &config, double cd){//, const CConfig &config)const {
  // cv::Mat tempDepth = *patch.getDepth();
  // cv::Mat depth = tempDepth(patch.getRoi());

  cv::Mat tempFeature = *patch.getFeature(4);
  cv::Rect tempRect = patch.getRoi();
  cv::Mat realFeature = tempFeature(patch.getRoi());

  double a = realFeature.at<double>(0,0) + realFeature.at<double>(tempRect.height,tempRect.width) - realFeature.at<double>(0,tempRect.width) - realFeature.at<double>(tempRect.height, 0);
  a /= tempRect.height;
  a /= tempRect.width;
  
  double sca = a;
  
  //cv::Mat showDepth = cv::Mat(tempDepth.rows, tempDepth.cols, CV_8UC1);

  //tempDepth.convertTo(showDepth, CV_8UC1, 255 / 1000);

  //cv::namedWindow("test");
  //cv::imshow("test", showDepth);
  //cv::waitKey(0);
  //cv::destroyWindow("test");

  //calc width and height scale
  //std::cout << depth.type() << " " << CV_8U << std::endl;
  //std::cout << config.p_height / 2 + 1 <<  config.p_width / 2 + 1 << std::endl;
  //  std::cout << "depth rows and cols " << depth.rows << " " << depth.cols << std::endl;
  //double centerDepth = depth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1) + config.mindist;
  cv::Point_<double> currentP = patch.getRelativePosition();

  //  double sca = tempDepth.at<ushort>(config.p_height / 2 + 1, config.p_width / 2 + 1);
  //  if(sca == 0)
  //  exit(-1);
  //std::cout << "current p before " << currentP << std::endl;
  //std::cout << sca << std::endl;

  //    currentP.x = currentP.x * 10;
  //  currentP.y *= 1000;
  //  currentP.x *= 1000;
  currentP.x *= sca;
  currentP.y *= sca;

  //std::cout << "current p " << currentP << std::endl;

  //std::cout << "kokomade" << std::endl;
  //  std::cout << "heknak go " << currentP << std::endl;
  patch.setRelativePosition(currentP);

  return 0;
}

