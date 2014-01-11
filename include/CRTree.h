#ifndef __CRTREE__
#define __CRTREE__

#include "CPatch.h"
#include "util.h"
#include "HaarLike.h"
#include <ctime>
#include <cmath>

// Auxilary structure
struct IntIndex {
  int val;
  unsigned int index;
  bool operator<(const IntIndex& a) const { return val<a.val; }
  static bool lessVal(const IntIndex& rLeft, const IntIndex& rRight) { return rLeft.val < rRight.val; }
};

class LeafNode
{
 public:
  LeafNode(){};
  ~LeafNode(){};

  //void show(int delay, int width, int height);
  //    void print()
  //    {
  //        std::cout << "Leaf " << vCenter.size() << " ";
  //        for(int i = 0; i < pfg.size(); i++)std::cout << pfg.at(i) << " ";
  //        std::cout << std::endl;
  //    }
  std::vector<float> pfg;
  //std::vector<std::vector<cv::Point> > vCenter; // per class per patch
  //std::vector<int> vClass;
  std::vector<std::vector<CParamset> > param; // per class per patch
};

class CTrainSet{
 public:
  CTrainSet(){}
  CTrainSet(std::vector<CPosPatch> &pos, 
	    std::vector<CNegPatch> &neg){posPatch = pos; negPatch = neg;}
  ~CTrainSet(){}

  int size()const{return posPatch.size() + negPatch.size();}

  std::vector<CPosPatch> posPatch;
  std::vector<CNegPatch> negPatch;
};

class CRTree 
{
 public:
  //constructor
 CRTree(int		min_s,	//min sample
	int		max_d,	//max depth of tree
	//	int		cp,	//number of center point
	CClassDatabase cDatabase/*,
				  boost::mt19937	randomGen*/	// random number seed
	)
   : min_samples(min_s), 
    max_depth(max_d), 
    num_leaf(0), 
    //    num_cp(cp), 
    classDatabase(cDatabase)
  {
    num_nodes = (int)pow(2.0, int(max_depth + 1)) - 1;

    // number of nodes x 7 matrix as vector
    treetable = new int[num_nodes * 11];
    // init treetable
    for(unsigned int i = 0; i< num_nodes * 11; ++i)
      treetable[i] = 0;

    leaf= new LeafNode[(int)pow(2.0, int(max_depth))];
  }

  CRTree(const char* filename, const char* datasetname, CConfig &conf);

  //destructor
  ~CRTree()
    {
      //std::cout << "destroy crtree" << std::endl;
      //if(treetable != NULL)
      delete [] treetable;
      //std::cout << "treetable!" << std::endl;
      //if(leaf != NULL)
      delete [] leaf;
      //std::cout << "released crtree" << std::endl;
    }

  // Set/Get functions
  unsigned int GetDepth() const {return max_depth;}
  //  unsigned int GetNumCenter() const {return num_cp;}

  // Regression
  const LeafNode* regression(CTestPatch &patch) const;

  // Training
  //void growTree(std::vector<std::vector<CPatch> > &TrData, int node, int depth, float pnratio, CConfig conf, boost::mt19937 gen,const std::vector<int> &defaultClass_);
  void growTree(std::vector<CPosPatch> &posPatch, std::vector<CNegPatch> &negPatch, int node, unsigned int depth, float pnratio, CConfig conf, const std::vector<int> &defaultClass_);
  //boost::mt19937 gen;
  bool optimizeTest(CTrainSet &SetA,
		    CTrainSet &SetB,
		    CTrainSet &trainSet,
		    int* test,
		    unsigned int iter,
		    unsigned int measure_mode,
		    int depth
		    );
  void generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c, int depth);

  void makeLeaf(CTrainSet &trainSet, float pnratio, int node);

  void evaluateTest(std::vector<std::vector<IntIndex> >& valSet, const int* test, CTrainSet &trainSet);
  void split(CTrainSet& SetA, CTrainSet& SetB, const CTrainSet& TrainSet, const std::vector<std::vector<IntIndex> >& valSet, int t);
  double distMean(const std::vector<CPosPatch>& SetA,
                  const std::vector<CPosPatch>& SetB);
  double InfGain(const CTrainSet& SetA, const CTrainSet& SetB);
  double calcEntropy(const CTrainSet &set);//, int negSize,int maxClass);
  double measureSet(const CTrainSet& SetA, const CTrainSet& SetB, unsigned int depth,int mode) {
    //double lamda = 1;
    if(mode == 1)
      return InfGain(SetA, SetB);
    else
      return distMean(SetA.posPatch, SetB.posPatch) * -1;
  }
  //void calcHaarLikeFeature(const cv::Mat &patch, const int* test, double &p1, double &p2) const;

  void setConfig(CConfig _conf){config = _conf;}
  
  // IO functions
  bool saveTree(const char* filename) const;
  //    void showLeaves(int width, int height) const {
  //        for(unsigned int l=0; l<num_leaf; ++l)
  //            leaf[l].show(5000, width, height);
  //    }

 private:
  // Data structure
  // tree table
  // 2^(max_depth+1)-1 x 7 matrix as vector
  // column: leafindex x1 y1 x2 y2 channel thres
  // if node is not a leaf, leaf=-1
  int* treetable;

  // stop growing when number of patches is less than min_samples
  unsigned int	min_samples;
  // depth of the tree: 0-max_depth
  unsigned int	max_depth;
  // number of nodes: 2^(max_depth+1)-1
  unsigned int	num_nodes;
  // number of leafs
  unsigned int	num_leaf;
  // number of center points per patch
  //  unsigned int	num_cp;
  //leafs as vector
  LeafNode*	leaf;

  CConfig config;

  // depth of this tree
  unsigned int depth;

  static boost::lagged_fibonacci1279 genTree;
  int nclass;

  std::vector<int> defaultClass, containClass, containClassA, containClassB;

  std::vector<std::vector<CPosPatch> > patchPerClass;

  CClassDatabase classDatabase;

  //void normarizationByDepth(CPatch* patch,cv::Mat& rgb)const;//, const CConfig &config)const;

};

inline void CRTree::generateTest(int* test, unsigned int max_w, unsigned int max_h, unsigned int max_c, int depth) {
  //double lamda = (double) config.max_depth;
  boost::mt19937    gen2(static_cast<unsigned long>(time(NULL)) );

  boost::uniform_int<> dst( 0, INT_MAX );
  boost::variate_generator<boost::lagged_fibonacci1279&,
    boost::uniform_int<> > rand( genTree, dst );

  boost::uniform_real<> dst2( 0, 1 );
  boost::variate_generator<boost::lagged_fibonacci1279&,
    boost::uniform_real<> > rand2( genTree, dst2 );

  //config.learningMode = 2;
  //  std::cout << "learning mode : " << config.learningMode << std::endl;

  switch(config.learningMode){
  case 0:// rgbd
    if(rand2() > 0.3){//(1 - exp(-1 * (double)depth / (lamda/5))) > rand2()){
      //if(1){
      if(config.rgbFeature == 1){

	// rgb
	test[0] = rand() % max_w;
	test[1] = rand() % max_h;
	test[4] = rand() % max_w;
	test[5] = rand() % max_h;
	test[8] = rand() % (max_c - 1);

	test[2] = 0;
	test[3] = 0;
	test[6] = 0;
	test[7] = 0;
      }else if(config.rgbFeature != 1){

	int rgb = rand() % 4;

	test[8] = rgb;
	// caliculate haar-like features
	int angle = rand() % 360;
	int type = rand() % 7;
	int ratio = 50;//(rand() % 40) + 10;
	if(config.depthFeature == 2)
	  test[0] = angle;
	else
	  test[0] = 0;
	test[1] = type;
	test[2] = ratio;

	test[3] = ((rand() % 50) + 30);
	test[4] = ((rand() % 50) + 30);
	test[5] = rand() % (100 - test[3]);
	test[6] = rand() % (100 - test[4]);
	test[7] = -1;

      }

    }else{

      // depth
      test[8] = max_c - 1;
      cv::Rect rect1, rect2;

      // caliculate haar-like features
      int angle = rand() % 360;
      int type = rand() % 7;
      int ratio = 50;//(rand() % 40) + 10;

      if(config.depthFeature == 2)
	test[0] = angle;
      else
	test[0] = 0;
      test[1] = type;
      test[2] = ratio;

      test[3] = ((rand() % 50) + 30);
      test[4] = ((rand() % 50) + 30);
      test[5] = rand() % (100 - test[3]);
      test[6] = rand() % (100 - test[4]);
      test[7] = -1;
    }
    break;
  case 1:
    {
      // depth
      test[8] = max_c - 1;

      // caliculate haar-like features
      int angle = rand() % 360;
      int type = rand() % 5;
      int ratio = (rand() % 60) + 20;

      test[0] = angle;
      test[1] = type;
      test[2] = ratio;

      test[3] = -1;
      test[4] = -1;
      test[5] = -1;
      test[6] = -1;
      test[7] = -1;
      break;
    }
  case 2:
    {
      // rgb
      if(config.rgbFeature == 1){

	// rgb
	test[0] = rand() % max_w;
	test[1] = rand() % max_h;
	test[4] = rand() % max_w;
	test[5] = rand() % max_h;
	test[8] = rand() % (max_c - 1);

	test[2] = 0;
	test[3] = 0;
	test[6] = 0;
	test[7] = 0;
      }else if(config.rgbFeature != 1){
	/* int rgb = rand() % 4; */

	/* test[8] = rgb; */
	/* // caliculate haar-like features */
	/* int angle = rand() % 360; */
	/* int type = rand() % 7; */
	/* int ratio = (rand() % 40) + 10; */
	/* if(config.depthFeature == 2) */
	/*   test[0] = angle; */
	/* else */
	/*   test[0] = 0; */
	/* test[1] = type; */
	/* test[2] = ratio; */

	/* test[3] = -1; */
	/* test[4] = -1; */
	/* test[5] = -1; */
	/* test[6] = -1; */
	/* test[7] = -1; */
	int rgb = rand() % 4;

	test[8] = rgb;
	// caliculate haar-like features
	int angle = rand() % 360;
	int type = rand() % 7;
	int ratio = 50;//(rand() % 40) + 10;
	if(config.depthFeature == 2)
	  test[0] = angle;
	else
	  test[0] = 0;
	test[1] = type;
	test[2] = ratio;

	test[3] = ((rand() % 50) + 30);
	test[4] = ((rand() % 50) + 30);
	test[5] = rand() % (100 - test[3]);
	test[6] = rand() % (100 - test[4]);
	test[7] = -1;

      }
      break;
    }
  default:
    std::cout << "error! can't set learning mode!" << std::endl;
    break;
  }

     /* std::cout << "evaluated test" << std::endl; */
     /* for(int i = 0; i < 9; ++i) */
     /*     std::cout << test[i] << " "; */
     /* std::cout << std::endl; */

}


#endif
