from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent


INSTALL_REQUIRES = [
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
    "clip",
    "lpips",
]


def discover_packages():
    packages = ["text2img"]

    for init_file in sorted(ROOT.rglob("__init__.py")):
        package_path = init_file.parent.relative_to(ROOT)
        if package_path == Path(".") or "__pycache__" in package_path.parts:
            continue
        packages.append("text2img." + ".".join(package_path.parts))

    return packages


setup(
    name="evolve-guidance-text2img",
    version="0.1.0",
    description="Text-to-image Stein guidance experiments for Evolve Guidance.",
    packages=discover_packages(),
    package_dir={"text2img": "."},
    include_package_data=True,
    package_data={
        "text2img": [
            "assets/*.pth",
            "prompts/*.json",
            "prompts/*.txt",
        ],
    },
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "text2img-approx-sd=text2img.runs.approx_sd:main",
            "text2img-approx-sdxl=text2img.runs.approx_sdxl:main",
            "text2img-grad-sd=text2img.runs.grad_sd:main",
            "text2img-grad-sdxl=text2img.runs.grad_sdxl:main",
        ],
    },
)
