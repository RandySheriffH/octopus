cmake_minimum_required(VERSION 3.1)
project(octopus VERSION 3.11.0)
add_executable(test test/main.cc)

include(ExternalProject)

ExternalProject_Add(tbb
                    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
                    GIT_TAG v2021.8.0
                    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbb-src
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbb-build
                    CMAKE_ARGS -DTBB_TEST=OFF -DTBBMALLOC_BUILD=OFF -DTBBMALLOC_PROXY_BUILD=OFF -DTBB_TEST=OFF -DTBB_OUTPUT_DIR_BASE=tbb
                    INSTALL_COMMAND "")

set(TBB_INC ${CMAKE_CURRENT_BINARY_DIR}/tbb-src/include)
set(TBB_BIN ${CMAKE_CURRENT_BINARY_DIR}/tbb-build/tbb_relwithdebinfo)
set(TBB_BIN_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/tbb-build/tbb_debug)
set(TBB_DEUBG "tbb12_debug")
set(TBB "tbb12")

target_include_directories(test PRIVATE include ${TBB_INC})
target_link_directories(test PRIVATE ${TBB_BIN} ${TBB_BIN_DEBUG})
target_link_libraries(test PRIVATE ${TBB} ${TBB_DEUBG})