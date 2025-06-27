cmake_minimum_required(VERSION 3.22)

project(SplatEditor)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# add_subdirectory(./libs/openvr-master)

include(cmake/common.cmake)

##############################################
## SUBPROJECT: Splat Editor
##############################################

set(TARGET_NAME SplatEditor) 

file(GLOB SplatEditor_SOURCES CONFIGURE_DEPENDS 
	src/*.*
	src/actions/*.*
	src/GPUPrefixSums/*.*
	src/GPUSorting/*.*
	src/gui/*.*
	src/render/*.*
	src/scene/*.*
	src/update/*.*
	src/loader/*.*
	src/writer/*.*
	include/*.*
)

if (WIN32)
	list(APPEND SplatEditor_SOURCES
		src/win32/GLRenderer_win32.cpp
		src/win32/OpenVRHelper.cpp
	)
elseif (UNIX)
	list(APPEND SplatEditor_SOURCES
		src/unix/GLRenderer_linux.cpp
	)
endif (WIN32)

add_executable(${TARGET_NAME} ${SplatEditor_SOURCES})

if (WIN32)
	# set workdir to ../$(SolutionDir), so that .cu files can be monitored/reloaded/compiled
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif (WIN32)

target_include_directories(${TARGET_NAME} PRIVATE 
	./include
	./src
)

target_compile_definitions(${TARGET_NAME}
    PRIVATE PROJECT_ROOT_DIR=\"${CMAKE_SOURCE_DIR}\")

#enable_language(CUDA)

# add some third-party libraries
ADD_CUDA(${TARGET_NAME})
ADD_OPENGL(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})
ADD_IMPLOT(${TARGET_NAME})
ADD_IMGUIZMO(${TARGET_NAME})
target_include_directories(${TARGET_NAME} PRIVATE ./include)
target_include_directories(${TARGET_NAME} PRIVATE ./libs)

target_link_libraries(${TARGET_NAME} 
    CUDA::cuda_driver      # cuda.lib
    CUDA::nvJitLink)       # nvJitLink.lib

# I give up, let's just include OpenVR like this.
target_include_directories(${TARGET_NAME} PRIVATE ./libs/openvr/headers)
if (WIN32)
	target_link_libraries(${TARGET_NAME} ../libs/openvr/lib/win64/openvr_api)

    # OpenVR API DLL
    # This is the OpenVR API DLL that we need to copy next to the built .exe
    set(OPENVR_DLL
        "${CMAKE_SOURCE_DIR}/libs/openvr/bin/win64/openvr_api.dll")

    # Copy the OpenVR API DLL to the output directory after build
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENVR_DLL}"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        COMMENT "POST BUILD: copying openvr_api.dll")
else()
	add_compile_definitions(NO_OPENVR)
endif (WIN32)

if (UNIX)
	find_package(TBB REQUIRED)
	target_link_libraries(${TARGET_NAME} TBB::tbb)
endif (UNIX)

##############################################
## SOME FINAL STUFF
##############################################

# set startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SplatEditor)