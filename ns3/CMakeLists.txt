cmake_minimum_required(VERSION 3.16)
project(video_qoe_ns3 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE VQM_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE VQM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

add_executable(run_scenario ${VQM_SOURCES} ${VQM_HEADERS})
target_include_directories(run_scenario PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# ns-3/5G-LENA linkage to be added later.
message(STATUS "Configured run_scenario (stub executable)")
