# step37-infertest — Step-3.7-Flash 单卡推理引擎（CUDA，sm_89 调优目标）
# 依赖：CUDA Toolkit（nvcc）、cuBLAS；vendored: llama.cpp ggml-cuda MMQ 子集 + ggml 头
cmake_minimum_required(VERSION 3.24)
project(step37-infertest LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89")     # RTX 4080 SUPER；其他卡改这里
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# ---- vendored llama.cpp ggml-cuda MMQ 子集（INT8 TC GEMM / quantize）----
file(GLOB VEND_CUDA_SOURCES CONFIGURE_DEPENDS
    "vendor/ggml-cuda/*.cu"
    "vendor/ggml-cuda/template-instances/*.cu")
add_library(ggml_cuda_mmq STATIC ${VEND_CUDA_SOURCES})
target_include_directories(ggml_cuda_mmq PRIVATE
    vendor                      # 源内含 "ggml-cuda/common.cuh" 相对包含
    vendor/ggml-include)
target_compile_definitions(ggml_cuda_mmq PRIVATE GGML_CUDA_FA GGML_CUDA_FA_F16_F16=1 GGML_CUDA_FA_F16_Q4_0=0 GGML_CUDA_FA_F16_Q4_1=0 GGML_CUDA_FA_F16_Q5_0=0 GGML_CUDA_FA_F16_Q5_1=0 GGML_CUDA_FA_F16_Q8_0=0 GGML_CUDA_FA_F16_BF16=0 GGML_CUDA_FA_Q4_0_F16=0 GGML_CUDA_FA_Q4_0_Q4_0=0 GGML_CUDA_FA_Q4_0_Q4_1=0 GGML_CUDA_FA_Q4_0_Q5_0=0 GGML_CUDA_FA_Q4_0_Q5_1=0 GGML_CUDA_FA_Q4_0_Q8_0=0 GGML_CUDA_FA_Q4_0_BF16=0 GGML_CUDA_FA_Q4_1_F16=0 GGML_CUDA_FA_Q4_1_Q4_0=0 GGML_CUDA_FA_Q4_1_Q4_1=0 GGML_CUDA_FA_Q4_1_Q5_0=0 GGML_CUDA_FA_Q4_1_Q5_1=0 GGML_CUDA_FA_Q4_1_Q8_0=0 GGML_CUDA_FA_Q4_1_BF16=0 GGML_CUDA_FA_Q5_0_F16=0 GGML_CUDA_FA_Q5_0_Q4_0=0 GGML_CUDA_FA_Q5_0_Q4_1=0 GGML_CUDA_FA_Q5_0_Q5_0=0 GGML_CUDA_FA_Q5_0_Q5_1=0 GGML_CUDA_FA_Q5_0_Q8_0=0 GGML_CUDA_FA_Q5_0_BF16=0 GGML_CUDA_FA_Q5_1_F16=0 GGML_CUDA_FA_Q5_1_Q4_0=0 GGML_CUDA_FA_Q5_1_Q4_1=0 GGML_CUDA_FA_Q5_1_Q5_0=0 GGML_CUDA_FA_Q5_1_Q5_1=0 GGML_CUDA_FA_Q5_1_Q8_0=0 GGML_CUDA_FA_Q5_1_BF16=0 GGML_CUDA_FA_Q8_0_F16=0 GGML_CUDA_FA_Q8_0_Q4_0=0 GGML_CUDA_FA_Q8_0_Q4_1=0 GGML_CUDA_FA_Q8_0_Q5_0=0 GGML_CUDA_FA_Q8_0_Q5_1=0 GGML_CUDA_FA_Q8_0_Q8_0=0 GGML_CUDA_FA_Q8_0_BF16=0 GGML_CUDA_FA_BF16_F16=0 GGML_CUDA_FA_BF16_Q4_0=0 GGML_CUDA_FA_BF16_Q4_1=0 GGML_CUDA_FA_BF16_Q5_0=0 GGML_CUDA_FA_BF16_Q5_1=0 GGML_CUDA_FA_BF16_Q8_0=0 GGML_CUDA_FA_BF16_BF16=0)
set_property(TARGET ggml_cuda_mmq PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(ggml_cuda_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 --extended-lambda>
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mf16c>)   # 头内 CPU 辅助（f16 表）走 AVX2/F16C

# ---- strata kernels（CPU lane / rope / norm / mmvq / router / moe_combine）----
add_library(strata_kernels STATIC
    src/kernels/cuda/native_gr_norm.cu
    src/kernels/cuda/native_rope.cu
    src/kernels/cuda/native_mmvq.cu
    src/kernels/cuda/router_sigmoid.cu
    src/kernels/cuda/elementwise.cu
    src/kernels/cuda/shared_expert.cu
    src/kernels/cuda/iq_kernels.cu
    src/kernels/cuda/strata_unused_stubs.cu
    src/kernels/cuda/quantize_act.cu
    src/kernels/cuda/bf16_gemv.cu
    src/kernels/cuda/native_bf16.cu
    src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/rope.cu
    src/kernels/cuda/gr.cu
    src/kernels/cuda/ple.cu
    src/kernels/cpu/expert.cpp
    src/kernels/cpu/expert_layout.cpp
    src/kernels/cpu/q2_avx2.cpp
    src/kernels/cpu/native_expert.cpp
    src/kernels/cpu/iq_avx512.cpp
    src/kernels/cpu/iq_avx2.cpp
    src/kernels/cpu/kq_avx512.cpp
    src/kernels/cpu/iq_avx512_one.cpp)
target_include_directories(strata_kernels PUBLIC include vendor/ggml-include)
target_compile_definitions(strata_kernels PRIVATE STRATA_NATIVE_EXPERTS=1)
target_compile_options(strata_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mf16c -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl
        -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq -mbmi2 -mbmi>)   # 多 ISA 全集（运行时 CPUID 守卫，Zen2 不进 AVX512 分支）
target_link_libraries(strata_kernels PRIVATE CUDA::cudart)

# ---- prefill MMQ 包装层 ----
add_library(prefill_mmq STATIC
    src/prefill/moe_mmq.cu
    src/prefill/ggml_cuda_host.cu
    src/prefill/gemm.cu
    src/prefill/kernels.cu)
target_include_directories(prefill_mmq PRIVATE include vendor/ggml-include vendor/ggml-cuda)
target_link_libraries(prefill_mmq PRIVATE ggml_cuda_mmq ggml_core CUDA::cudart CUDA::cublas)

# ---- 引擎 ----
add_executable(step37-load
    src/step37/step37_model.cu
    src/step37/step37_aux.cu
    src/step37/step37_attn.cu
    src/step37/step37_gguf.cpp
    src/step37/step37_main.cpp)
target_include_directories(step37-load PRIVATE src/step37 include vendor/ggml-include)
target_link_libraries(step37-load PRIVATE strata_kernels prefill_mmq ggml_cuda_mmq ggml_core
    CUDA::cudart CUDA::cublas CUDA::cuda_driver)
add_library(vendor_shim STATIC src/vendor_shim/ggml_abort_stub.cpp)

# ---- ggml 核心（CPU 编译；ggml-cuda 后端 API 面 + gguf/quant 数学）----
file(GLOB GGML_CORE_SOURCES CONFIGURE_DEPENDS
    "vendor/ggml-core/*.c" "vendor/ggml-core/*.cpp"
    "vendor/ggml-cpu/*.c" "vendor/ggml-cpu/*.cpp"
    "vendor/ggml-cpu/arch/x86/*.c" "vendor/ggml-cpu/arch/x86/*.cpp")
list(LENGTH GGML_CORE_SOURCES GGML_CORE_N)
message(STATUS "GGML_CORE_N=${GGML_CORE_N}")
add_library(ggml_core STATIC ${GGML_CORE_SOURCES})
target_include_directories(ggml_core PRIVATE vendor/ggml-include vendor/ggml-core vendor/ggml-cpu vendor/ggml-cpu)
target_include_directories(ggml_core PRIVATE vendor/ggml-cpu)  # amx/llamafile 子目录头
target_compile_definitions(ggml_core PRIVATE _GNU_SOURCE STRATA_NATIVE_EXPERTS=1)
target_compile_options(ggml_core PRIVATE
    -mavx2 -mf16c -mfma
    -mavx512f -mavx512bw -mavx512dq -mavx512vl
    -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq)   # x86 arch 量化实现需 ISA 全集（运行时 CPUID 分派）
target_link_libraries(ggml_cuda_mmq PRIVATE vendor_shim)
target_compile_options(step37-load PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
