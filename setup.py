import os

from setuptools import setup


def get_extension_config():
    build_cuda_ext = os.environ.get("ACORN_BUILD_CUDA_EXT") == "1"
    if not build_cuda_ext:
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:
        raise RuntimeError(
            "ACORN_BUILD_CUDA_EXT=1 requires torch to be importable at build time. "
            "Build in the target environment with --no-build-isolation."
        ) from exc

    compile_args = {
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math"],
    }
    extensions = [
        CUDAExtension(
            name="acorn.cuda_ext._connected_components_cuda",
            sources=[
                "acorn/cuda_ext/connected_components.cpp",
                "acorn/cuda_ext/connected_components_kernel.cu",
            ],
            extra_compile_args=compile_args,
        ),
        CUDAExtension(
            name="acorn.cuda_ext._reverse_topological_dp_cuda",
            sources=[
                "acorn/cuda_ext/reverse_topological_dp.cpp",
                "acorn/cuda_ext/reverse_topological_dp_kernel.cu",
            ],
            extra_compile_args=compile_args,
        ),
        CUDAExtension(
            name="acorn.cuda_ext._trace_selected_paths_cuda",
            sources=[
                "acorn/cuda_ext/trace_selected_paths.cpp",
                "acorn/cuda_ext/trace_selected_paths_kernel.cu",
            ],
            extra_compile_args=compile_args,
        ),
    ]
    return extensions, {"build_ext": BuildExtension}


ext_modules, cmdclass = get_extension_config()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)