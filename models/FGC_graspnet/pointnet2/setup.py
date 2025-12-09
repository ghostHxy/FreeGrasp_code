from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")

INCLUDE_DIRS = [
    os.path.join(ROOT, "_ext_src", "include"),
    os.path.join(CONDA_PREFIX, "lib/python3.10/site-packages/nvidia/cuda_runtime/include"),
    os.path.join(CONDA_PREFIX, "targets/x86_64-linux/include"),
    os.path.join(CONDA_PREFIX, "include"),
]

LIB_DIRS = [
    os.path.join(CONDA_PREFIX, "lib"),
    os.path.join(CONDA_PREFIX, "lib64"),
]

print("Using include dirs:", INCLUDE_DIRS)
print("Using library dirs:", LIB_DIRS)

_ext_src_root = "_ext_src"
_ext_sources = glob.glob(f"{_ext_src_root}/src/*.cpp") + glob.glob(f"{_ext_src_root}/src/*.cu")

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs=INCLUDE_DIRS,
            library_dirs=LIB_DIRS,
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2", "--expt-relaxed-constexpr"]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
