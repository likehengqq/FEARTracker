cmake_minimum_required(VERSION 3.10)
project(fear_rk3588_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(RKNN_API_PATH "" CACHE PATH "Path to Rockchip rknn runtime package")
find_path(
  RKNN_INCLUDE_DIR
  rknn_api.h
  HINTS
    ${RKNN_API_PATH}/include
    $ENV{RKNN_API_PATH}/include
    /usr/include
    /usr/local/include
)
find_library(
  RKNN_RT_LIB
  rknnrt
  HINTS
    ${RKNN_API_PATH}/lib
    ${RKNN_API_PATH}/lib64
    $ENV{RKNN_API_PATH}/lib
    $ENV{RKNN_API_PATH}/lib64
    /usr/lib
    /usr/local/lib
    /usr/lib/aarch64-linux-gnu
)

if(NOT RKNN_INCLUDE_DIR)
  message(FATAL_ERROR "rknn_api.h not found. Set -DRKNN_API_PATH=/path/to/rknn/runtime")
endif()

if(NOT RKNN_RT_LIB)
  message(FATAL_ERROR "librknnrt.so not found. Set -DRKNN_API_PATH=/path/to/rknn/runtime")
endif()

add_executable(fear_rk3588_demo fear_rk3588_demo.cpp)
target_include_directories(fear_rk3588_demo PRIVATE ${RKNN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fear_rk3588_demo PRIVATE ${RKNN_RT_LIB} ${OpenCV_LIBS})
