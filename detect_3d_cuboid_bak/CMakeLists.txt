cmake_minimum_required(VERSION 3.0.0)

project(detect_cuboid)
message("build for ${PROJECT_NAME}")
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE relwithdebinfo)

MESSAGE("Build type: ${CMAKE_BUILD_TYPE} for ${PROJECT_NAME}")


find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    include_directories("/usr/include/eigen3")
endif()

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(LINE_LSD_NAME line_lbd)
set(YOLO_NAME YOLOv8)
set(LINE_LSD_PATH ${PROJECT_SOURCE_DIR}/../${LINE_LSD_NAME})
set(YOLO_PATH ${PROJECT_SOURCE_DIR}/../${YOLO_NAME})
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../onnxruntime-linux-x64-1.16.3")


#ADD_SUBDIRECTORY(${LINE_LSD_PATH} ${LINE_LSD_PATH}/build/line_lbd)
#ADD_SUBDIRECTORY(${YOLO_PATH} ${YOLO_PATH}/build)

ADD_SUBDIRECTORY(${LINE_LSD_PATH} ${PARENT_SOURCE_DIR}/build/${LINE_LSD_NAME})
ADD_SUBDIRECTORY(${YOLO_PATH} ${PARENT_SOURCE_DIR}/build/${YOLO_NAME})

set(CUBOID_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/object_3d_util.cpp
    ${PROJECT_SOURCE_DIR}/src/matrix_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/box_proposal_detail.cpp
)


#include_directories(
#    ${OpenCV_INCLUDE_DIRS}
#    ${LINE_LSD_PATH}/include
#    ${YOLO_PATH}/include
#    ${ONNXRUNTIME_DIR}/include
#    ${PROJECT_SOURCE_DIR}/include
#)


add_library(${PROJECT_NAME} SHARED ${CUBOID_SOURCE_FILES})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${PROJECT_SOURCE_DIR}/include
                           ${LINE_LSD_PATH}/include
                           ${YOLO_PATH}/include
                           ${EIGEN3_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS}  
                           ${ONNXRUNTIME_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBRARIES}
                      line_lsd
                      yolov8
)


#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#  message(STATUS "dir='${dir}'")
#endforeach()