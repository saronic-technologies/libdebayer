cmake_minimum_required(VERSION 3.15)

project(libdebayer-cpp)

set(libdebayer-cpp_VERSION_MAJOR 0)
set(libdebayer-cpp_VERSION_MINOR 1)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(libdebayer REQUIRED)

find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75) # Adjust architecture

# Lib
cuda_add_library(debayercpp SHARED
    src/debayer_cpp.cu
)

include_directories(debayercpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(debayercpp libdebayer::debayer)

set_target_properties(debayercpp PROPERTIES
    VERSION ${libdebayer-cpp_VERSION_MAJOR}.${libdebayer-cpp_VERSION_MINOR}
    SOVERSION ${libdebayer-cpp_VERSION_MAJOR}
)

set_target_properties(debayercpp PROPERTIES PUBLIC_HEADER include/debayer_cpp.h)

include(GNUInstallDirs)

install(TARGETS debayercpp
        EXPORT libdebayercppConfig
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION include/libdebayercpp)

install(EXPORT libdebayercppConfig
        FILE libdebayercppConfig.cmake
        NAMESPACE libdebayercpp::
        DESTINATION lib/cmake/)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        "libdebayercppConfigVersion.cmake"
        VERSION ${libdebayer-cpp_VERSION_MAJOR}.${libdebayer-cpp_VERSION_MINOR}
        COMPATIBILITY AnyNewerVersion)

configure_file(libdebayercpp.pc.in libdebayercpp.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/libdebayercpp.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)



