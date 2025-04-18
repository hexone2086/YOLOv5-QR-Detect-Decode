cmake_minimum_required(VERSION 3.8)
project(qr_detector)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

# Set OpenCV path (using opencv-mobile)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please check OpenCV_DIR or install OpenCV.")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

# 设置 ONNX Runtime 的根目录
set(ONNXRUNTIME_ROOT ${CMAKE_SOURCE_DIR}/onnxruntime)
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Find ZBar
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZBAR REQUIRED zbar)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/onnxruntime/include
    # ${ONNXRuntime_INCLUDE_DIRS}
    ${ZBAR_INCLUDE_DIRS}
)

# Add executable
add_executable(qr_detector main.cc)

# Link libraries
target_link_libraries(qr_detector
    ${OpenCV_LIBS}
    # ${ONNXRuntime_LIBRARIES}
    ${ZBAR_LIBRARIES}
    onnxruntime
)
