from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import os
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")

# 自动优先使用 torch 对应的 CUDA_HOME
CUDA_PATH = CUDA_HOME or CONDA_PREFIX
print("Detected CUDA PATH:", CUDA_PATH)

INCLUDE_DIRS = [
    os.path.join(ROOT, "_ext_src", "include"),
    os.path.join(CUDA_PATH, "include"),
    os.path.join(CONDA_PREFIX, "include"),
    os.path.join(CONDA_PREFIX, "lib/python3.10/site-packages/nvidia/cuda_runtime/include"),
    os.path.join(CONDA_PREFIX, "targets/x86_64-linux/include"),
]

LIB_DIRS = [
    os.path.join(CUDA_PATH, "lib64"),
    os.path.join(CONDA_PREFIX, "lib"),
    os.path.join(CONDA_PREFIX, "lib64"),
]

# 添加 symlink fallback 因应 CUDA12+ thrust 安装新位置
THRUST_HEADER = os.path.join(CUDA_PATH, "include/thrust")
if not os.path.exists(THRUST_HEADER):
    conda_thrust = os.path.join(CONDA_PREFIX, "targets/x86_64-linux/include/thrust")
    if os.path.exists(conda_thrust):
        print("Using thrust from conda:", conda_thrust)
        INCLUDE_DIRS.append(os.path.join(CONDA_PREFIX, "targets/x86_64-linux/include"))

print("Using include dirs:", INCLUDE_DIRS)
print("Using library dirs:", LIB_DIRS)

_ext_src_root = "_ext_src"
_ext_sources = glob.glob(f"{_ext_src_root}/src/*.cpp") + glob.glob(f"{_ext_src_root}/src/*.cu")

extra_compile_args = {
    "cxx": ["-O2"],
    "nvcc": [
        "-O2",
        "--expt-relaxed-constexpr",
        "-Xcompiler", "-fPIC"
    ],
}

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs=INCLUDE_DIRS,
            library_dirs=LIB_DIRS,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
