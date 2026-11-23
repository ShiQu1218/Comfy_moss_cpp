cmake_minimum_required(VERSION 3.28)
project(moss_lokr_cache LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
find_package(CUDAToolkit REQUIRED)
set(ENGINE "${CMAKE_CURRENT_SOURCE_DIR}/../.upstream/src")
add_executable(moss-lokr-cache lokr_cache.cpp normalized_metadata.cpp
  ${ENGINE}/models/lora/lora_file.cpp lokr_merge.cu
  ${ENGINE}/weights/safetensors.cpp ${ENGINE}/weights/file_reader.cpp
  ${ENGINE}/weights/supported_model.cpp ${ENGINE}/weights/quant/quant_codec.cpp
  ${ENGINE}/weights/model_index.cpp ${ENGINE}/weights/gguf.cpp ${ENGINE}/util/json.cpp
  ${ENGINE}/util/regular_file.cpp ${ENGINE}/util/replaceable_file.cpp ${ENGINE}/util/xxh64.cpp)
target_include_directories(moss-lokr-cache PRIVATE ${ENGINE} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(moss-lokr-cache PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_compile_options(moss-lokr-cache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
if(MSVC)
  target_compile_options(moss-lokr-cache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  target_compile_options(moss-lokr-cache PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2;/Gy;/fp:precise;/utf-8>)
  target_link_options(moss-lokr-cache PRIVATE /OPT:REF)
else()
  target_compile_options(moss-lokr-cache PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2;-ffunction-sections;-fdata-sections;-ffp-contract=off;-fno-fast-math>)
  target_link_options(moss-lokr-cache PRIVATE -Wl,--gc-sections)
  if(WIN32)
    target_link_options(moss-lokr-cache PRIVATE -static -municode)
  endif()
endif()
