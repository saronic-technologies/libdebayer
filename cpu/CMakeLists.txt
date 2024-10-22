# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(DebayerCPU)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Source files
set(SOURCES
    main.cpp
    cpu_debayer.cpp
    threadpool.cpp
    cpu_kernel.cpp
)

# Add executable
add_executable(debayer_cpu ${SOURCES})

# Link libraries
target_link_libraries(debayer_cpu ${OpenCV_LIBS})
