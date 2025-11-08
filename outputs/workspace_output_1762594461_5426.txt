from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="resolveai-assistant",
    version="1.0.0",
    author="ResolveAI Contributors",
    author_email="contributors@resolveai.ai",
    description="Open Source AI Assistant for DaVinci Resolve Video Editing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/resolveai/resolveai-assistant",
    project_urls={
        "Bug Tracker": "https://github.com/resolveai/resolveai-assistant/issues",
        "Documentation": "https://docs.resolveai.ai",
        "Source Code": "https://github.com/resolveai/resolveai-assistant",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Display",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.2.0",
            "mkdocstrings>=0.22.0",
        ],
        "aws": [
            "boto3>=1.28.0",
            "botocore>=1.31.0",
        ],
        "gcp": [
            "google-cloud-storage>=2.10.0",
        ],
        "azure": [
            "azure-storage-blob>=12.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resolveai=resolveai.cli:main",
            "resolveai-server=resolveai.server:main",
            "resolveai-plugin=resolveai.plugin:main",
        ],
    },
    include_package_data=True,
    package_data={
        "resolveai": [
            "config/*.yaml",
            "models/*.onnx",
            "assets/*.png",
            "assets/*.ico",
        ],
    },
    keywords="video editing, ai, davinci resolve, machine learning, computer vision, assistant",
    zip_safe=False,
)