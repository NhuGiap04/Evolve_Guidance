from setuptools import setup, find_packages

setup(
    name="seg",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numba==0.60.0",
        "numpy==2.0.0",
        "scipy==1.14.0",
        "matplotlib",
        "ml-collections==0.1.1",
        "absl-py==2.1.0",
        "diffusers==0.32.2",
        "accelerate==1.3.0",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "inflect==7.5.0",
        "pydantic==2.10.6",
        "transformers==4.48.2",
        "timm==1.0.14",
        "huggingface-hub==0.28.1",
        "fairscale==0.4.13",
        "hpsv2==1.2.0",
        "protobuf<4",
        "clip",
        "lpips",
    ],
    extras_require={
        "tce": [
            "image-diversity @ git+https://github.com/fibarrola/image_diversity.git",
        ],
        "geneval": [
            "mmdet==2.28.2",
            "mmcv-full==1.7.1",
            "open-clip-torch==2.20.0",
            "clip-benchmark==1.4.0",
            "pycocotools==2.0.6",
            "pandas>=1.5.0",
            "pillow>=9.0.0",
        ]
    }
)
