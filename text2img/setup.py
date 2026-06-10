from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent


def read_requirements():
    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.exists():
        return []

    requirements = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


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
    install_requires=read_requirements(),
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
