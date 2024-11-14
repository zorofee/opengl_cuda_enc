cmake_minimum_required(VERSION 3.0)
project (REIDRENDER)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)  # 设置 C++ 标准为 C++17
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++17")

find_package(OpenCV REQUIRED)
include_directories(/usr/local/include /usr/local/cuda ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDES})
include_directories(VideoCodec/NvDec VideoCodec/NvEnc VideoCodec/Interface Render VideoCodec/CudaProc libmov/include)

find_package(OpenGL REQUIRED)
add_definitions(${OPENGL_DEFINITIONS})

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)
set(EGL_LIBS ${OPENGL_opengl_LIBRARY}  ${OPENGL_egl_LIBRARY})

find_library(CUVID_LIB nvcuvid PATHS /data/julang/demo/Nvidia-Video-Codec/nvenc_11.0.10/Lib/linux/stubs/aarch64)
find_library(NVENCODEAPI_LIB nvidia-encode PATHS /data/julang/demo/Nvidia-Video-Codec/nvenc_11.0.10/Lib/linux/stubs/aarch64)

aux_source_directory(VideoCodec/NvDec DEC)
aux_source_directory(VideoCodec/NvEnc ENC)
aux_source_directory(Render RENDER)
aux_source_directory(libmov/source MOV)
set(CUDA_SOURCES VideoCodec/CudaProc/ColorSpace.cu)  # Add all your .cu files here

link_directories(/usr/local/lib /usr/lib)

set(EXECUTABLE_OUTPUT_PATH ./)
add_compile_options(-g  -fpermissive -std=c++17) 
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
cuda_add_library(render SHARED ${DEC} ${ENC} ${RENDER} ${CUDA_SOURCES} ${MOV})
target_link_libraries(render avutil avformat avcodec swscale swresample nvcuvid nvidia-encode ${OpenCV_LIBS} ${CUDA_LIBRARIES} cuda ${EGL_LIBS})

cuda_add_executable(demo test/test.cpp)
target_link_libraries(demo render)
