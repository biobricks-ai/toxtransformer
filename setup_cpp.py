from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Set CUDA architectures for T4 (7.5) and A100 (8.0)
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5;8.0'

setup(
    name='toxtransformer_cuda',
    ext_modules=[
        CUDAExtension(
            name='toxtransformer_cuda',
            sources=['csrc/tensor_builder.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
