cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_FLAGS "-std=c++20")
Set(CMAKE_BUILD_TYPE Release)

project(Neural_Network)

include_directories(third_party/mnist/include)

add_subdirectory(src)

add_executable(mnist examples/mnist_trainer.cpp)
target_include_directories(mnist PRIVATE src)
target_link_libraries(mnist PRIVATE Neural_Network)
