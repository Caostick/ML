cmake_minimum_required (VERSION 3.1)

option(ML_TESTS "Enable build of tests" OFF)

set(PROJECT_NAME ML)
set(CPP_VERSION 17)

project(${PROJECT_NAME})

set_property(GLOBAL PROPERTY ML_TMP_SRC_LIST)
function(ML_GATHER_FILES DIRNAME DIR)
	FILE(GLOB ML_TMP_FILES ../${PROJECT_NAME}/${DIR}/*.h ../${PROJECT_NAME}/${DIR}/*.cpp)
	source_group(${DIRNAME} FILES ${ML_TMP_FILES})
	get_property(ML_LOCAL_SRC_LIST GLOBAL PROPERTY ML_TMP_SRC_LIST)	
	set_property(GLOBAL PROPERTY ML_TMP_SRC_LIST ${ML_LOCAL_SRC_LIST} ${ML_TMP_FILES})
endfunction()

set_property(GLOBAL PROPERTY TMP_SRC_LIST)

# === BEGIN FILES ===
ML_GATHER_FILES("Include\\ML" Include/ML)
ML_GATHER_FILES("Source" Source)
# === END FILES ===

get_property(ML_SRC_LIST GLOBAL PROPERTY ML_TMP_SRC_LIST)
add_library (${PROJECT_NAME} STATIC ${ML_SRC_LIST})

include_directories(Include/)

set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "ML")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD ${CPP_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")

if(ML_TESTS)
add_subdirectory(Tests)
endif()