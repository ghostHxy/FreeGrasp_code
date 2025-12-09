# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os
import site

ROOT = os.path.dirname(os.path.abspath(__file__))

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

# 获取 conda 环境中的 CUDA 头文件路径
def get_cuda_include_paths():
    """查找 conda 环境中的 CUDA 头文件路径"""
    include_paths = []
    
    # 获取 site-packages 路径
    site_packages = site.getsitepackages()
    if not site_packages:
        # 如果没有找到，尝试使用当前环境的 site-packages
        import sys
        site_packages = [f for f in sys.path if 'site-packages' in f]
    
    for sp in site_packages:
        # 查找 nvidia/cuda_runtime/include
        nvidia_cuda_path = os.path.join(sp, "nvidia", "cuda_runtime", "include")
        if os.path.exists(nvidia_cuda_path):
            include_paths.append(nvidia_cuda_path)
        
        # 查找 triton 的 CUDA 头文件
        triton_cuda_path = os.path.join(sp, "triton", "backends", "nvidia", "include")
        if os.path.exists(triton_cuda_path):
            include_paths.append(triton_cuda_path)
    
    return include_paths

# 添加 CUDA 头文件路径到编译参数
cuda_include_paths = get_cuda_include_paths()
local_include = os.path.join(ROOT, _ext_src_root, "include")

# 构建编译参数
extra_compile_args_cxx = ["-O2", "-I{}".format(local_include)]
extra_compile_args_nvcc = ["-O2", "-I{}".format(local_include)]

# 添加 CUDA 头文件路径
for path in cuda_include_paths:
    extra_compile_args_cxx.append("-I{}".format(path))
    extra_compile_args_nvcc.append("-I{}".format(path))

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": extra_compile_args_cxx,
                "nvcc": extra_compile_args_nvcc,
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)