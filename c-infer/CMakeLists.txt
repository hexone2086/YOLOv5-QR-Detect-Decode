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

# Set ONNX Runtime path (关键修改点)
set(onnxruntime_DIR ${CMAKE_SOURCE_DIR}/onnxruntime/lib64/cmake/onnxruntime)
find_package(onnxruntime REQUIRED CONFIG)  # 显式指定使用 CONFIG 模式

if(NOT onnxruntime_FOUND)
    message(FATAL_ERROR "ONNX Runtime not found. Please check onnxruntime_DIR or install ONNX Runtime.")
endif()

message(STATUS "ONNX Runtime include path: ${ONNXRuntime_INCLUDE_DIRS}")

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
    ${CMAKE_SOURCE_DIR}/onnxruntime/lib64/libonnxruntime.so
    ${ZBAR_LIBRARIES}
)
