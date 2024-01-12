cmake_minimum_required(VERSION 3.5)
project(PocketNN)
set(EXEC_TARGET PocketNN)
set(POCKETNN_DIR PocketNN)

# Find and add the include directories for libcurl library
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Add all the source files to the POCKETNN_SRC variable
file(GLOB POCKETNN_SRC "${POCKETNN_DIR}/*.cpp")

# Add the executable target and link against the libcurl library
add_executable(${EXEC_TARGET} ${POCKETNN_SRC})
target_include_directories(${EXEC_TARGET} PRIVATE ${POCKETNN_DIR})
target_link_libraries(${EXEC_TARGET} ${CURL_LIBRARIES})

# Copy the dataset directory to the build directory
file(COPY ${CMAKE_SOURCE_DIR}/${POCKETNN_DIR}/dataset DESTINATION ${CMAKE_BINARY_DIR})

# Set the C++ standard to C++11
set_target_properties(${EXEC_TARGET} PROPERTIES CXX_STANDARD 11)
