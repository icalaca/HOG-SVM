cmake_minimum_required(VERSION 3.5)
project(HOG)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES HOG.cpp Training.cpp Training.h Utility.cpp Utility.h)
add_executable(HOG ${SOURCE_FILES})

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLU REQUIRED)
find_package(OpenCV 3 REQUIRED )

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${OpenCV_LIBS})
