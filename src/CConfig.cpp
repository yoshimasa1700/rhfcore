#include "CConfig.h"

using namespace boost;
using namespace boost::property_tree;

int CConfig::loadConfig(const char* filename)
{
  
  std::cout << std::endl << "loaded config" << std::endl << std::endl;
  read_xml(filename, pt);

  try{
    // load tree path
    treepath = pt.get<std::string>("root.treepath");
    std::cout << "tree path is " << treepath << std::endl;
    // load number of tree
    ntrees = pt.get<int>("root.ntree");
    std::cout << "number of tree is " << ntrees << std::endl;


    // load patch width
    p_width = pt.get<int>("root.pwidth");
    std::cout << "patch width is " << p_width << std::endl;
    // load patch height
    p_height = pt.get<int>("root.pheight");
    std::cout << "patch height is " << p_height << std::endl;


    // load scale factor for output imae
    patchRatio = pt.get<double>("root.patchratio");
    std::cout << "patch per tree is " << patchRatio << std::endl;
    // load patch stride
    stride = pt.get<int>("root.stride");
    std::cout << "patch stride is " << stride << std::endl;
    // load train image num per tree
    imagePerTree = pt.get<int>("root.trainimagepertree");
    std::cout << "image num per tree is " << imagePerTree << std::endl;
    // load min sample num
    min_sample = pt.get<int>("root.minsample");
    std::cout << "min sample num is " << min_sample << std::endl;
    // load max depth num
    max_depth = pt.get<int>("root.maxdepth");
    std::cout << "max depth of tree is " << max_depth << std::endl;
    // load ratio of pos patch number and neg patch number
    pnRatio = pt.get<double>("root.posnegpatchratio");
    std::cout << "ratio of pos patch number and neg patch number is " << pnRatio << std::endl;
    // not work
    acPatchRatio = pt.get<double>("root.activepatchratio");
    std::cout << "ratio of active patch is " << acPatchRatio << std::endl;
    // load number of trials
    nOfTrials = pt.get<int>("root.numberOfTrials");
    std::cout << "number of trials is " << nOfTrials << std::endl;


    // learning mode
    // 1:depth 2:rgb 0:rgbd
    learningMode = pt.get<int>("root.learningmode");
    std::cout << "learning mode is " << learningMode << std::endl;
    // rgb feature select
    // 0: haar-like, 1: HOG, 2: rotated haar-like
    rgbFeature = pt.get<int>("root.rgbfeature");
    std::cout << "rgb feature is " << rgbFeature << std::endl;
    // depth feature select
    // 0: haar-like, 1: HOG, 2: rotated haar-like
    depthFeature = pt.get<int>("root.depthfeature");
    std::cout << "depth feature is " << depthFeature << std::endl;
 

    // train pos data folders
    trainpath = pt.get<std::string>("root.trainposdata.rootpath");
    traindatafile = pt.get<std::string>("root.trainposdata.folderlist");
    traindatalist = pt.get<std::string>("root.trainposdata.imagelist");
    std::cout << "train pos root path is " << trainpath << std::endl;
    std::cout << "train pos folder list is " << traindatafile << std::endl;
    std::cout << "train pos image list is " << traindatalist << std::endl;
    // train neg data folders
    negDataPath = pt.get<std::string>("root.trainnegdata.rootpath");
    negFolderList = pt.get<std::string>("root.trainnegdata.folderlist");
    negDataList = pt.get<std::string>("root.trainnegdata.imagelist");
    std::cout << "train neg root path is " << negDataPath << std::endl;
    std::cout << "train neg folder list is " << negFolderList << std::endl;
    std::cout << "train neg image list is " << negDataList << std::endl;
    // test data folders
    testPath = pt.get<std::string>("root.testdata.rootpath");
    testData = pt.get<std::string>("root.testdata.folderlist");
    testdatalist = pt.get<std::string>("root.testdata.imagelist");
    std::cout << "test data root path is " << testPath << std::endl;
    std::cout << "test data folder list is " << testData << std::endl;
    std::cout << "test data image list is " << testdatalist << std::endl;


    // offset of tree num
    off_tree = pt.get<int>("root.offtree");
    std::cout << "offset of tree num is " << off_tree << std::endl;
    // class database name
    classDatabaseName = pt.get<std::string>("root.classdatabasename");
    std::cout << "class database name is " << classDatabaseName << std::endl;
    // min distance
    mindist = pt.get<int>("root.mindistance");
    std::cout << "min dist is " << mindist << std::endl;
    // max distance
    maxdist = pt.get<int>("root.maxdistance");
    std::cout << "max dist is " << maxdist << std::endl;

    poseEC = pt.get<int>("root.poseestimatecircle");
    std::cout << "pose estimation circle is " << poseEC << std::endl;
    showVote = pt.get<int>("root.showvote");
    std::cout << "show vote is " << poseEC << std::endl;
    
  }

  catch(ptree_bad_path bp){
    std::cout << "bad property tree path! " << bp.path<std::string>() << std::endl;
    exit(-1);
  }
  catch(ptree_bad_data bd){
    std::cout << "bad property tree data! " << bd.data<std::string>() << std::endl;
    exit(-1);
  }

  std::cout << std::endl;

  return 0;
}

