cmake_minimum_required(VERSION 3.0.0)
project(line_lsd)
message("build for ${PROJECT_NAME}")
#set(CMAKE_CXX_STANDARD 14)

STRING(REGEX MATCH "-std=c\\+\\+[0-9]+" result ${CMAKE_CXX_FLAGS})
IF(NOT ${result} MATCHES "-std=c\\+\\+[0-9]+")
    set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
ENDIF()

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

find_package(OpenCV REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(LINE_LSD_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/binary_descriptor_matcher.cpp
    ${PROJECT_SOURCE_DIR}/src/binary_descriptor.cpp
    ${PROJECT_SOURCE_DIR}/src/draw.cpp
    ${PROJECT_SOURCE_DIR}/src/line_lbd_allclass.cpp
    ${PROJECT_SOURCE_DIR}/src/lsd.cpp
    ${PROJECT_SOURCE_DIR}/src/LSDDetector.cpp
)

#include_directories(
#    ${PROJECT_SOURCE_DIR}/include
#    ${OpenCV_INCLUDE_DIRS}
#)


add_library(${PROJECT_NAME} SHARED ${LINE_LSD_SOURCE_FILES})

#target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
target_include_directories(${PROJECT_NAME} PRIVATE
                           ${PROJECT_SOURCE_DIR}/include
                           ${OpenCV_INCLUDE_DIRS}  
)
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBRARIES}
                      ${EIGEN3_LIBS}
)
