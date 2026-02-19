cmake_minimum_required(VERSION 3.10)
project(point-morphology)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(FETCHCONTENT_QUIET FALSE)

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -march=native") #  -ffast-math seems to not work

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
   GIT_TAG        LeonardEyer-fix-dynamic-tree-sorted
)

FetchContent_Declare(
   ssg
   GIT_REPOSITORY https://codeberg.org/LeonardEyer/ssg.git
   GIT_TAG        v0.3
)

set(NANOFLANN_BUILD_EXAMPLES OFF)

FetchContent_MakeAvailable(polyscope libigl happly nanoflann ssg)


# Header-only interface target
add_library(pointmorph INTERFACE)

target_include_directories(pointmorph INTERFACE
    ${CMAKE_SOURCE_DIR}/inc
    ${happly_SOURCE_DIR}
)

target_link_libraries(pointmorph INTERFACE
    polyscope
    Eigen3::Eigen
    nanoflann::nanoflann
    ssg::ssg_lib
)

file(GLOB SOURCES "src/*.cpp")

add_executable(point-morphology ${SOURCES})
target_link_libraries(point-morphology PRIVATE pointmorph)

add_subdirectory(tests)
