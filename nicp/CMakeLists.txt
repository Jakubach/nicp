# Include the subdirectories
ADD_SUBDIRECTORY(nicp)
IF(BUILD_NICP_VIEWER)
  ADD_SUBDIRECTORY(nicp_viewer)
ENDIF(BUILD_NICP_VIEWER)
IF(BUILD_NICP_TEST)
  ADD_SUBDIRECTORY(nicp_test)
ENDIF(BUILD_NICP_TEST)