cmake_minimum_required(VERSION 3.0.0)

project(orb_object_slam)
message("build for ${PROJECT_NAME}")
#set(CMAKE_CXX_STANDARD 14)


STRING(REGEX MATCH "-std=c\\+\\+[0-9]+" result ${CMAKE_CXX_FLAGS})
IF(NOT ${result} MATCHES "-std=c\\+\\+[0-9]+")
    set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
ENDIF()


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)

IF(PCL_COMPILE_OPTIONS)
    MESSAGE(STATUS "  PCL_COMPILE_OPTIONS = ${PCL_COMPILE_OPTIONS}")
else()
    message("PCL_COMPILE_OPTIONS no")
ENDIF(PCL_COMPILE_OPTIONS)

set(DBOW2_PATH ${PROJECT_SOURCE_DIR}/../Thirdparty/DBoW2)
set(G2O_PATH ${PROJECT_SOURCE_DIR}/../Thirdparty/g2o)

set(LINE_LSD_NAME line_lbd)
set(YOLO_NAME YOLOv8)
set(DETECT_3d_CUBOID_NAME detect_3d_cuboid)

set(LINE_LSD_PATH ${PROJECT_SOURCE_DIR}/../${LINE_LSD_NAME})
set(YOLO_PATH ${PROJECT_SOURCE_DIR}/../${YOLO_NAME})
set(CUBOID_PATH ${PROJECT_SOURCE_DIR}/../${DETECT_3d_CUBOID_NAME})
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../onnxruntime-linux-x64-1.16.3")

ADD_SUBDIRECTORY(${G2O_PATH} ${G2O_PATH}/build)
ADD_SUBDIRECTORY(${DBOW2_PATH} ${DBOW2_PATH}/build)


#ADD_SUBDIRECTORY(${LINE_LSD_PATH} ${LINE_LSD_PATH}/build)
#ADD_SUBDIRECTORY(${YOLO_PATH} ${YOLO_PATH}/build)
#ADD_SUBDIRECTORY(${CUBOID_PATH} ${CUBOID_PATH}/build)
ADD_SUBDIRECTORY(${CUBOID_PATH} ${PARENT_SOURCE_DIR}/build/${DETECT_3d_CUBOID_NAME})

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE relwithdebinfo)

MESSAGE("Build type: ${CMAKE_BUILD_TYPE} for ${PROJECT_NAME}")
MESSAGE("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} in ${PROJECT_NAME}")
MESSAGE("CMAKE_C_FLAGS: ${CMAKE_C_FLAGS} in ${PROJECT_NAME}")
MESSAGE("CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE} in ${PROJECT_NAME}")
MESSAGE("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE} in ${PROJECT_NAME}")

set(ORB_OBJ_SLAM_FILES 
    ${PROJECT_SOURCE_DIR}/src/Converter.cc
    ${PROJECT_SOURCE_DIR}/src/Frame.cc
    ${PROJECT_SOURCE_DIR}/src/FrameDrawer.cc
    ${PROJECT_SOURCE_DIR}/src/g2o_Object.cpp
    ${PROJECT_SOURCE_DIR}/src/Initializer.cc
    ${PROJECT_SOURCE_DIR}/src/KeyFrame.cc
    ${PROJECT_SOURCE_DIR}/src/KeyFrameDatabase.cc
    ${PROJECT_SOURCE_DIR}/src/LocalMapping.cc
    ${PROJECT_SOURCE_DIR}/src/LoopClosing.cc
    ${PROJECT_SOURCE_DIR}/src/Map.cc
    ${PROJECT_SOURCE_DIR}/src/MapDrawer.cc
    ${PROJECT_SOURCE_DIR}/src/MapObject.cc
    ${PROJECT_SOURCE_DIR}/src/MapPoint.cc
    ${PROJECT_SOURCE_DIR}/src/Optimizer.cc
    ${PROJECT_SOURCE_DIR}/src/ORBextractor.cc
    ${PROJECT_SOURCE_DIR}/src/ORBmatcher.cc
    ${PROJECT_SOURCE_DIR}/src/Parameters.cpp
    ${PROJECT_SOURCE_DIR}/src/PnPsolver.cc
    ${PROJECT_SOURCE_DIR}/src/Sim3Solver.cc
    ${PROJECT_SOURCE_DIR}/src/System.cc
    ${PROJECT_SOURCE_DIR}/src/Tracking_util.cc
    ${PROJECT_SOURCE_DIR}/src/Tracking.cc
    ${PROJECT_SOURCE_DIR}/src/Viewer.cc
)

#include_directories(
#    ${PROJECT_SOURCE_DIR}/include
#    ${OpenCV_INCLUDE_DIRS}
#    ${EIGEN_INCLUDE_DIR}
#    ${Pangolin_INCLUDE_DIRS}
#    ${PCL_INCLUDE_DIRS}
#    ${DBOW2_PATH}
#    ${G2O_PATH}
#)

add_library(${PROJECT_NAME} SHARED ${ORB_OBJ_SLAM_FILES})

#target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${PROJECT_SOURCE_DIR}/include
                           ${PROJECT_SOURCE_DIR}/../
                           ${OpenCV_INCLUDE_DIRS}
                           ${EIGEN_INCLUDE_DIR}
                           ${Pangolin_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS}
                           ${DBOW2_PATH}
                           ${G2O_PATH}
                           ${CUBOID_PATH}/include
                           ${YOLO_PATH}/include
                           ${YOLO_PATH}/bytetrack/include
                           ${LINE_LSD_PATH}/include
                           ${ONNXRUNTIME_DIR}/include
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBS}
                      ${Pangolin_LIBRARIES}
                      ${PCL_LIBRARIES}
                      DBoW2
                      g2o
#                     ${ONNXRUNTIME_DIR}/include
                      line_lsd
                      yolov8
                      detect_cuboid
                      -lboost_serialization
                      -lcrypto
)
