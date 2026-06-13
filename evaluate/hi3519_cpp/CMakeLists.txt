cmake_minimum_required(VERSION 3.10)
project(fear_hi3519_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1) CPU 头部数值自检（仅依赖标准库，始终可编译，可在任意 PC 上验证）
add_executable(test_fear_head test_fear_head.cpp)

# 2) 板端 demo：需要 OpenCV；NNIE 骨干需 -DUSE_HISVP=ON 并提供海思 SVP SDK
option(USE_HISVP "启用海思 SVP NNIE 骨干后端（需要 SVP SDK）" OFF)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
  add_executable(fear_hi3519_demo fear_hi3519_demo.cpp)
  target_include_directories(fear_hi3519_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(fear_hi3519_demo PRIVATE ${OpenCV_LIBS})
  if(USE_HISVP)
    target_compile_definitions(fear_hi3519_demo PRIVATE USE_HISVP)
    set(HISVP_API_PATH "" CACHE PATH "海思 SVP SDK 路径")
    target_include_directories(fear_hi3519_demo PRIVATE ${HISVP_API_PATH}/include .)
    # 链接 NNIE / SVP 相关库，请按你的 SDK 适配，例如：
    # target_link_libraries(fear_hi3519_demo PRIVATE nnie svp_npu mpi securec)
  endif()
else()
  message(STATUS "未找到 OpenCV：仅构建 test_fear_head。板端请安装 OpenCV 后再构建 fear_hi3519_demo。")
endif()
