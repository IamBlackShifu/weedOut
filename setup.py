from setuptools import setup, find_packages

setup(
    name="weedout",
    version="0.1.0",
    description="NLP-based cyberbullying detection and profiling system for Twitter-like platforms",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "nltk>=3.8.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.0.0",
        "emoji>=2.10.0",
    ],
)
