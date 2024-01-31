cmake_minimum_required(VERSION 3.0.0)
project(cube_slam)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

#SET(CMAKE_BUILD_TYPE Debug)

message("build for ${PROJECT_NAME}")
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")

# inference by onnx if use_onnx else opencv.dnn
ADD_DEFINITIONS(-DUSE_ONNX=true )
#ADD_DEFINITIONS(-DEIGEN_MAX_ALIGN_BYTES=32)

# Compiler specific options for gcc
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}    -Wall  -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -Wall  -O3 -march=native")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}   -Wall  -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}   -Wall  -O3 -march=native")


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address -fno-stack-protector -fno-var-tracking   ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address -fno-stack-protector -fno-var-tracking ")
#set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(DBOW2_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2)
set(G2O_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/g2o)
set(LINE_LSD_PATH ${PROJECT_SOURCE_DIR}/line_lbd)
set(YOLO_PATH ${PROJECT_SOURCE_DIR}/YOLOv8)
set(CUBOID_PATH ${PROJECT_SOURCE_DIR}/detect_3d_cuboid)
set(ORB_OBJECT_SLAM_PATH ${PROJECT_SOURCE_DIR}/orb_object_slam)
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3")

set(PARENT_SOURCE_DIR ${PROJECT_SOURCE_DIR})


#ADD_SUBDIRECTORY(${DBOW2_PATH} ${DBOW2_PATH}/build)
#ADD_SUBDIRECTORY(${G2O_PATH} ${G2O_PATH}/build)
#ADD_SUBDIRECTORY(${LINE_LSD_PATH} ${LINE_LSD_PATH}/build)
#ADD_SUBDIRECTORY(${YOLO_PATH} ${YOLO_PATH}/build)
#ADD_SUBDIRECTORY(${CUBOID_PATH} ${CUBOID_PATH}/build)

#ADD_SUBDIRECTORY(${ORB_OBJECT_SLAM_PATH} ${ORB_OBJECT_SLAM_PATH}/build)
ADD_SUBDIRECTORY(${ORB_OBJECT_SLAM_PATH})

MESSAGE("Build type: ${CMAKE_BUILD_TYPE} for ${PROJECT_NAME}")
MESSAGE("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS} in ${PROJECT_NAME}")
MESSAGE("CMAKE_C_FLAGS: ${CMAKE_C_FLAGS} in ${PROJECT_NAME}")
MESSAGE("CMAKE_C_FLAGS_RELEASE: ${CMAKE_C_FLAGS_RELEASE} in ${PROJECT_NAME}")
MESSAGE("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE} in ${PROJECT_NAME}")

find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    include_directories("/usr/include/eigen3")
endif()
#add_definitions(-DENABLE_PRECOMPILED_HEADERS=OFF)
#message("EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

find_package(OpenCV 4.8 REQUIRED)


# build examples 
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/bin)

add_executable(mono_kitti ${PROJECT_SOURCE_DIR}/Examples/mono_kitti.cc)
target_include_directories(mono_kitti PRIVATE 
                           ${PROJECT_SOURCE_DIR}
                           ${PROJECT_SOURCE_DIR}/line_lbd/include 
                           ${PROJECT_SOURCE_DIR}/YOLOv8/include 
                           ${PROJECT_SOURCE_DIR}/detect_3d_cuboid/include 
                           ${PROJECT_SOURCE_DIR}/orb_object_slam/include
                           ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2
)
target_link_libraries(mono_kitti line_lsd yolov8 detect_cuboid orb_object_slam)


add_executable(detect_lines ${PROJECT_SOURCE_DIR}/Examples/detect_lines.cc)
target_include_directories(detect_lines PRIVATE 
                           ${PROJECT_SOURCE_DIR}/line_lbd/include 
)
target_link_libraries(detect_lines  line_lsd)

add_executable(detect_3d_cuboid ${PROJECT_SOURCE_DIR}/Examples/detect_3d_cuboid.cc)
target_include_directories(detect_3d_cuboid PRIVATE 
                           ${PROJECT_SOURCE_DIR}/detect_3d_cuboid/include 
                           ${PROJECT_SOURCE_DIR}/YOLOv8/include 
                           ${PROJECT_SOURCE_DIR}/YOLOv8/bytetrack/include
                           ${PROJECT_SOURCE_DIR}/line_lbd/include 
                           ${EIGEN3_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS} 
                           ${ONNXRUNTIME_DIR}/include

)
target_link_libraries(detect_3d_cuboid 
                        line_lsd 
                        yolov8 
                        detect_cuboid 
                        ${OpenCV_LIBRARIES}
                        ${EIGEN3_LIBS}
                        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)


add_executable(detect_lines_3d_cuboid ${PROJECT_SOURCE_DIR}/Examples/detect_lines_3d_cuboid.cc)
target_include_directories(detect_lines_3d_cuboid PRIVATE 
                            ${PROJECT_SOURCE_DIR}/line_lbd/include 
                            ${PROJECT_SOURCE_DIR}/detect_3d_cuboid/include 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/include 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/bytetrack/include
                            ${EIGEN3_INCLUDE_DIR}
                            ${OpenCV_INCLUDE_DIRS} 
                            ${ONNXRUNTIME_DIR}/include

)
target_link_libraries(detect_lines_3d_cuboid 
                        line_lsd 
                        detect_cuboid
                        yolov8
                        ${OpenCV_LIBRARIES}
                        ${EIGEN3_LIBS}
                        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)


add_executable(yolov8_detect_segment ${PROJECT_SOURCE_DIR}/Examples/yolov8_detect_segment.cc)
target_include_directories(yolov8_detect_segment PRIVATE 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/include 
                            ${OpenCV_INCLUDE_DIRS}  
                            ${ONNXRUNTIME_DIR}/include
)
target_link_libraries(yolov8_detect_segment 
                            yolov8
                            ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
                            ${OpenCV_LIBRARIES}
                        )

add_executable(yolov8_detect_track ${PROJECT_SOURCE_DIR}/Examples/yolov8_detect_track.cc)
target_include_directories(yolov8_detect_track PRIVATE 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/include 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/bytetrack/include
                            ${OpenCV_INCLUDE_DIRS}  
                            ${ONNXRUNTIME_DIR}/include
                            ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(yolov8_detect_track 
                            yolov8
                            ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
                            ${EIGEN3_LIBS}
                            ${OpenCV_LIBRARIES}
                        )

add_executable(yolov8_segment_track ${PROJECT_SOURCE_DIR}/Examples/yolov8_segment_track.cc)
target_include_directories(yolov8_segment_track PRIVATE 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/include 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/bytetrack/include
                            ${OpenCV_INCLUDE_DIRS}  
                            ${ONNXRUNTIME_DIR}/include
                            ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(yolov8_segment_track 
                            yolov8
                            ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
                            ${EIGEN3_LIBS}
                            ${OpenCV_LIBRARIES}
                        )


add_executable(detect_2d_lines_3d_cuboid ${PROJECT_SOURCE_DIR}/Examples/detect_2d_lines_3d_cuboid.cc)
target_include_directories(detect_2d_lines_3d_cuboid PRIVATE 
                            ${PROJECT_SOURCE_DIR}/YOLOv8/include 
                            ${PROJECT_SOURCE_DIR}/line_lbd/include 
                            ${PROJECT_SOURCE_DIR}/detect_3d_cuboid/include 
                            ${OpenCV_INCLUDE_DIRS}  
                            ${ONNXRUNTIME_DIR}/include
                            ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(detect_2d_lines_3d_cuboid 
                        line_lsd 
                        yolov8 
                        detect_cuboid
                        ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
                        ${EIGEN3_LIBS}
                        ${OpenCV_LIBRARIES}
                    )

