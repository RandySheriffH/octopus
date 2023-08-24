cmake_minimum_required(VERSION 3.1)
project(octopus VERSION 3.11.0)
add_executable(test_oct test/test.cc)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(ExternalProject)

message("CMAKE_BUILD_TYPE" ${CMAKE_BUILD_TYPE})
message("CMAKE_CONFIGURATION_TYPES" ${CMAKE_CONFIGURATION_TYPES})

if(WIN32)
  set(TBB_BINARY "tbb12.dll")
else()
  set(TBB_BINARY "libtbb12.so")
endif()

ExternalProject_Add(intel_tbb
                    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
                    GIT_TAG v2021.8.0
                    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbb-src
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbb-build
                    CMAKE_ARGS -DTBB_TEST=OFF -DTBBMALLOC_BUILD=OFF -DTBBMALLOC_PROXY_BUILD=OFF -DTBB_TEST=OFF -DTBB_OUTPUT_DIR_BASE=tbb
                    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/tbb-build/tbb_relwithdebinfo/${TBB_BINARY} ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/${TBB_BINARY})

set(TBB_INC ${CMAKE_CURRENT_BINARY_DIR}/tbb-src/include)
set(TBB_BIN ${CMAKE_CURRENT_BINARY_DIR}/tbb-build/tbb_relwithdebinfo)

if(WIN32)
  set(TBB "tbb12")
else()
  set(TBB "tbb")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_include_directories(test_oct PRIVATE include ${TBB_INC})
target_link_directories(test_oct PRIVATE ${TBB_BIN})

if(WIN32)
  target_link_libraries(test_oct PRIVATE ${TBB})
else()
  find_package(Threads REQUIRED)
  target_link_libraries(test_oct PRIVATE ${TBB} Threads::Threads)
endif()

add_dependencies(test_oct intel_tbb)