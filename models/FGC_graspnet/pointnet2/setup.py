# Copyright (c) Facebook, Inc.
# MIT License

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CUDA_HOME = os.environ.get("CONDA_PREFIX", None)

# Include CUDA toolkit headers from conda installation
cuda_include = os.path.join(CUDA_HOME, "targets", "x86_64-linux", "include")
cuda_lib = os.path.join(CUDA_HOME, "targets", "x86_64-linux", "lib")

_ext_src_root = "_ext_src"
_ext_sources = glob.glob(f"{_ext_src_root}/src/*.cpp") + \
               glob.glob(f"{_ext_src_root}/src/*.cu")

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs=[
                f"{ROOT}/{_ext_src_root}/include",
                cuda_include,
            ],
            library_dirs=[cuda_lib],
            extra_compile_args={
                "cxx": [
                    "-O2",
                    f"-I{ROOT}/{_ext_src_root}/include",
                    "-std=c++17",
                    "-D_GLIBCXX_USE_CXX11_ABI=0"
                ],
                "nvcc": [
                    "-O2",
                    f"-I{ROOT}/{_ext_src_root}/include",
                    f"-I{cuda_include}",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                    "-D_GLIBCXX_USE_CXX11_ABI=0",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__"
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
