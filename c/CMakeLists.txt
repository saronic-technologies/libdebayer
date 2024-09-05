cmake_minimum_required(VERSION 3.15)

project(libdebayer)

set(libdebayer_VERSION_MAJOR 0)
set(libdebayer_VERSION_MINOR 1)

find_package(CUDA REQUIRED)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75) # Adjust architecture

# Lib
cuda_add_library(debayer SHARED
    src/debayer.cu
    src/debayer_kernel.cu
    src/debayer_kernel.h
)

include_directories(debayer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(debayer PROPERTIES
    VERSION ${libdebayer_VERSION_MAJOR}.${libdebayer_VERSION_MINOR}
    SOVERSION ${libdebayer_VERSION_MAJOR}
)
set_target_properties(debayer PROPERTIES PUBLIC_HEADER include/debayer.h)

include(GNUInstallDirs)

install(TARGETS debayer
        EXPORT libdebayerConfig
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION include/libdebayer)

install(EXPORT libdebayerConfig
        FILE libdebayerConfig.cmake
        NAMESPACE libdebayer::
        DESTINATION lib/cmake/)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        "libdebayerConfigVersion.cmake"
        VERSION ${libdebayer_VERSION_MAJOR}.${libdebayer_VERSION_MINOR}
        COMPATIBILITY AnyNewerVersion)

configure_file(libdebayer.pc.in libdebayer.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/libdebayer.pc
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
