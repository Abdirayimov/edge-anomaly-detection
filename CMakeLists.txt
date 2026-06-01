cmake_minimum_required(VERSION 3.22)
project(eanom VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerSettings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_TOOLS "Build CLI tools (benchmark, fit_padim_stats)" ON)
option(BUILD_DEEPSTREAM "Build the DeepStream pipeline (requires DeepStream SDK + GStreamer)" ON)
option(BUILD_TRT_PADIM "Build the TRT-based PaDiM detector" ON)
option(BUILD_TESTS "Build unit tests" OFF)

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio video)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

if(BUILD_TRT_PADIM)
    find_package(TensorRT)
    if(NOT TensorRT_FOUND)
        message(WARNING "TensorRT not found; disabling PaDiM TRT detector")
        set(BUILD_TRT_PADIM OFF)
    endif()
endif()

if(BUILD_DEEPSTREAM)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
        gstreamer-1.0
        gstreamer-app-1.0
        gobject-2.0
        glib-2.0
    )
    find_package(DeepStream 8.0)
    if(NOT DeepStream_FOUND)
        message(WARNING "DeepStream SDK not found; disabling DeepStream pipeline")
        set(BUILD_DEEPSTREAM OFF)
    endif()
endif()

file(GLOB_RECURSE EANOM_SOURCES
    "${CMAKE_SOURCE_DIR}/src/config/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/detectors/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/ensemble/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/roi/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/events/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/trt/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/pipeline/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/overlay/*.cpp"
)

if(NOT BUILD_TRT_PADIM)
    list(FILTER EANOM_SOURCES EXCLUDE REGEX "padim_detector\\.cpp$")
    list(FILTER EANOM_SOURCES EXCLUDE REGEX "feature_extractor\\.cpp$")
    list(FILTER EANOM_SOURCES EXCLUDE REGEX "trt_engine\\.cpp$")
endif()
if(NOT BUILD_DEEPSTREAM)
    list(FILTER EANOM_SOURCES EXCLUDE REGEX "deepstream_pipeline\\.cpp$")
endif()

add_library(eanom STATIC ${EANOM_SOURCES})
target_include_directories(eanom
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(eanom
    PUBLIC
        CUDA::cudart
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
)
if(BUILD_TRT_PADIM)
    target_include_directories(eanom PUBLIC ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(eanom PUBLIC TensorRT::nvinfer TensorRT::nvonnxparser)
    target_compile_definitions(eanom PUBLIC EANOM_HAVE_TRT=1)
endif()
if(BUILD_DEEPSTREAM)
    target_include_directories(eanom PUBLIC ${DeepStream_INCLUDE_DIRS})
    target_link_libraries(eanom PUBLIC
        DeepStream::nvds_meta
        DeepStream::nvbufsurface
        PkgConfig::GST
    )
    target_compile_definitions(eanom PUBLIC EANOM_HAVE_DEEPSTREAM=1)
endif()
set_target_properties(eanom PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(EXISTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
    add_executable(eanom_video src/main.cpp)
    target_link_libraries(eanom_video PRIVATE eanom)
endif()

if(BUILD_TOOLS AND EXISTS "${CMAKE_SOURCE_DIR}/tools/benchmark.cpp")
    add_executable(eanom_benchmark tools/benchmark.cpp)
    target_link_libraries(eanom_benchmark PRIVATE eanom)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TARGET eanom_video)
    install(TARGETS eanom_video RUNTIME DESTINATION bin)
endif()
if(TARGET eanom_benchmark)
    install(TARGETS eanom_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY configs/ DESTINATION share/eanom/configs)

message(STATUS "")
message(STATUS "eanom ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:              ${CUDAToolkit_VERSION}")
message(STATUS "  OpenCV:            ${OpenCV_VERSION}")
message(STATUS "  TRT PaDiM:         ${BUILD_TRT_PADIM}")
message(STATUS "  DeepStream:        ${BUILD_DEEPSTREAM}")
message(STATUS "  Build tools:       ${BUILD_TOOLS}")
message(STATUS "  Build tests:       ${BUILD_TESTS}")
message(STATUS "")
