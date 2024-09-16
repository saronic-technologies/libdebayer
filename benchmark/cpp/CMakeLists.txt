cmake_minimum_required(VERSION 3.15)

project(test_debayer)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Modify if yours is different.  This one is Ubuntu:
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib/x86_64-linux-gnu/cmake/opencv4/)

# Find OpenCV package
find_package(OpenCV 4.2 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(CUDA REQUIRED)
# #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75) # Adjust architecture

find_package(libdebayer REQUIRED)
find_package(libdebayercpp REQUIRED)

# App
add_executable(test_debayer src/test_debayer.cpp)
target_link_libraries(test_debayer
    ${OpenCV_LIBS}
    libdebayercpp::debayercpp
    ${CUDA_LIBRARIES}
)
target_include_directories(test_debayer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

install(TARGETS test_debayer
        RUNTIME DESTINATION bin)
