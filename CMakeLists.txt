cmake_minimum_required(VERSION 3.10)
project(point-morphology)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


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

FetchContent_MakeAvailable(polyscope)
FetchContent_MakeAvailable(libigl)

target_link_libraries(point-morphology polyscope Eigen3::Eigen)


