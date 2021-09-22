cmake_minimum_required(VERSION 3.12)
project(tinypt VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "") # static library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "") # shared library
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "") # executable

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# third-party libraries
include_directories(third_party)
add_subdirectory(third_party)

# OpenMP
find_package(OpenMP REQUIRED)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# GoogleTest
option(TINYPT_ENABLE_TEST "" ON)
if (TINYPT_ENABLE_TEST)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Do not install GTest when `make install`
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)
endif ()

if (NOT DEFINED TINYPT_ENABLE_CUDA)
    # cuda not specified: enable cuda if found
    find_package(CUDA)
    set(TINYPT_ENABLE_CUDA ${CUDA_FOUND})
endif ()

if (TINYPT_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    add_compile_definitions(TINYPT_ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G")
    set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -DNDEBUG -O3 --use_fast_math")
endif ()

option(DCHECK_ALWAYS_ON "" OFF)
if (DCHECK_ALWAYS_ON)
    add_compile_definitions(DCHECK_ALWAYS_ON)
endif ()

# sources
add_subdirectory(src)

option(TINYPT_ENABLE_INSTALL "" ON)
if (TINYPT_ENABLE_INSTALL)
    # ref: https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html
    # install binaries
    install(TARGETS tinypt tinyobjloader
            EXPORT tinyptTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    # install headers
    install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")
    # write package version
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/tinyptConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY AnyNewerVersion
    )
    # write cmake config file
    install(EXPORT tinyptTargets
            FILE tinyptTargets.cmake
            NAMESPACE tinypt::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinypt)
    configure_package_config_file(src/tinypt/tinyptConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/tinyptConfig.cmake
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinypt)
    # install to cmake directory
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tinyptConfigVersion.cmake ${CMAKE_CURRENT_BINARY_DIR}/tinyptConfig.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinypt)
endif ()

# clang-format
file(GLOB_RECURSE SOURCES src/*.cpp src/*.h src/*.cu src/*.cuh)
add_custom_target(format COMMAND clang-format -i ${SOURCES})
