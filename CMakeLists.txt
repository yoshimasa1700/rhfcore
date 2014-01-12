# CMakeのバージョン
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(CMAKE_CXX_COMPILER c++)

# リリースビルド用オプション
SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O2 -fpermissive -fPIC -pg")
# デバッグビルド用オプション
SET(CMAKE_CXX_FLAGS_DEBUG "-g -fpermissive -fPIC")
# ビルドのタイプはリリース
SET(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

#set(CMAKE_CXX_FLAGS "$(CMAKE_CXX_FLAGS) -fpermissive")

SET(CMAKE_INSTALL_PREFIX=/usr/local)
# プロジェクト名
PROJECT(rhfcore)

FIND_PACKAGE(OpenCV)
FIND_PACKAGE(Boost)
find_package(OpenMP)
#find_package(OpenGL)
#find_package(GLUT)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# バージョン番号
SET(serial "1.0.0")
# 共有ライブラリのバージョン番号
SET(soserial "1")
# ビルドするためにインクルードするヘッダーファイルの場所
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include")

# 共有ライブラリとしてビルド
ADD_LIBRARY(rhfcore SHARED 
		      src/CConfig.cpp 
		      src/CRForest.cpp 
		      src/HoG.cpp 
		      src/CDataset.cpp 
		      src/CRTree.cpp 
		      src/util.cpp
		      src/HaarLike.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  TARGET_LINK_LIBRARIES(rhfcore ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
else(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  TARGET_LINK_LIBRARIES(rhfcore ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

# バージョン番号の適用
SET_TARGET_PROPERTIES(rhfcore PROPERTIES VERSION ${serial} SOVERSION ${soserial})

# 共有ライブラリのインストール指定
INSTALL(TARGETS rhfcore LIBRARY DESTINATION lib)
# 共有ライブラリ用ヘッダーファイルのインストール指定
INSTALL(FILES include/CConfig.h 
	      include/CDetectionResult.h 
	      include/CRForest.h 
	      include/HoG.h 
	      include/CDataset.h 
	      include/CPatch.h 
	      include/CRTree.h 
	      include/util.h 
	      include/HaarLike.h
	      DESTINATION include/rhfcore)
