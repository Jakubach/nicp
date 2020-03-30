CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(depth_camera_tracking)

SET(depth_camera_tracking_C_FLAGS)
SET(depth_camera_tracking_CXX_FLAGS)

# Set the output directory for the build executables and libraries
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${depth_camera_tracking_SOURCE_DIR}/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${depth_camera_tracking_SOURCE_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${depth_camera_tracking_SOURCE_DIR}/lib)

# Set search directory for looking for our custom CMake scripts to look for other libraries
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${depth_camera_tracking_SOURCE_DIR}/cmake_modules)

# Set some compiling flag
ADD_DEFINITIONS(-DUNIX)
MESSAGE(STATUS "Compiling on Unix")

SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse4.2") 
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -msse4.2") 
SET(depth_camera_tracking_C_FLAGS "${depth_camera_tracking_C_FLAGS} -Wall -W")
SET(depth_camera_tracking_CXX_FLAGS "${depth_camera_tracking_CXX_FLAGS} -Wall -W")

# Specifying compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${depth_camera_tracking_CXX_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${depth_camera_tracking_C_FLAGS}")

# Find Eigen3
FIND_PACKAGE(Eigen3 3.2.0 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
MESSAGE(STATUS "Compiling with Eigen3")
MESSAGE(STATUS "\tEIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})

# Find Opencv
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "Compiling with OpenCV")
MESSAGE(STATUS "\tOpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "\tOpenCV_LIBS: " ${OpenCV_LIBS})

# OpenGL is used in the draw actions for the different types, as well
# as for creating the GUI itself
FIND_PACKAGE(OpenGL REQUIRED)
INCLUDE_DIRECTORIES(${OPENGL_INCLUDE})

# Find Qt5
FIND_PACKAGE(Qt5 COMPONENTS Core Xml OpenGL Gui Widgets REQUIRED)
INCLUDE_DIRECTORIES(
  ${Qt5Core_INCLUDE_DIRS} ${Qt5Xml_INCLUDE_DIRS} ${Qt5Gui_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS}
)
SET(CMAKE_CXX_FLAGS "-fPIC")

# For building the GUI
FIND_PACKAGE(QGLViewer REQUIRED)
INCLUDE_DIRECTORIES(${QGLVIEWER_INCLUDE_DIR})

# Find Nicp
FIND_PACKAGE(Nicp REQUIRED)
INCLUDE_DIRECTORIES(${NICP_INCLUDE_DIR})
MESSAGE(STATUS "Compiling with Nicp")
MESSAGE(STATUS "\tNICP_INCLUDE_DIR: " ${NICP_INCLUDE_DIR})
MESSAGE(STATUS "\tNICP_LIBRARY: " ${NICP_LIBRARY})
MESSAGE(STATUS "\tNICP_VIEWER_LIBRARY: " ${NICP_VIEWER_LIBRARY})
  
# Set up the top-level include directories
INCLUDE_DIRECTORIES(${depth_camera_tracking_SOURCE_DIR})

# Add the executable
ADD_EXECUTABLE(nicp_depth_camera_tracking nicp_depth_camera_tracking.cpp)
TARGET_LINK_LIBRARIES(nicp_depth_camera_tracking
  ${NICP_LIBRARY} ${NICP_VIEWER_LIBRARY}
  ${OpenCV_LIBS}
  ${QGLVIEWER_LIBRARY} Qt5::Xml Qt5::OpenGL Qt5::Gui Qt5::Core Qt5::Widgets
  ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
)
