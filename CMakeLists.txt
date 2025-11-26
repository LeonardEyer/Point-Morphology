cmake_minimum_required(VERSION 3.10)
project(point-morphology)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(FETCHCONTENT_QUIET FALSE)

# Release build: optimize aggressively
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")


include_directories(inc)
file(GLOB SOURCES "src/*.cpp")

add_executable(point-morphology ${SOURCES})

include(FetchContent)

FetchContent_Declare(
   polyscope
   GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
   GIT_TAG        v2.5.0
)

FetchContent_Declare(
   libigl
   GIT_REPOSITORY https://github.com/libigl/libigl.git
   GIT_TAG        v2.6.0
)

FetchContent_Declare(
   happly
   GIT_REPOSITORY https://github.com/nmwsharp/happly.git
   GIT_TAG        v1.0.0
)

FetchContent_Declare(
   nanoflann
   GIT_REPOSITORY https://github.com/LeonardEyer/nanoflann.git
   GIT_TAG     LeonardEyer-fix-dynamic-tree-sorted
)

set(NANOFLANN_BUILD_EXAMPLES OFF)

FetchContent_MakeAvailable(polyscope libigl happly nanoflann)

target_include_directories(point-morphology PUBLIC ${happly_SOURCE_DIR})

target_link_libraries(point-morphology polyscope Eigen3::Eigen nanoflann::nanoflann)
