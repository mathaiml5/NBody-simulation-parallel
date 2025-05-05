cmake_minimum_required(VERSION 3.10)
project(nbody_simulation CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/parlaylib/include)

# Source files
set(SOURCES
    main.cpp
    methods.cpp
)

# Header files
set(HEADERS
    vector.h
    body.h
    octree.h
    octree.tpp
    fmm.h
    fmm.tpp
    bvh.h
    bvh.tpp
    methods.h
)

# Add executable
add_executable(nbody_sim ${SOURCES} ${HEADERS})
target_link_libraries(nbody_sim PRIVATE ${OpenMP_CXX_LIBRARIES})

# Add ParlayLib
add_subdirectory(parlaylib)
target_link_libraries(nbody_sim PRIVATE parlay)

# Install target
install(TARGETS nbody_sim DESTINATION bin)
