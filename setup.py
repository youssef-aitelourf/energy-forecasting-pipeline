from setuptools import setup, find_packages

setup(
    name="energy-forecasting-pipeline",
    version="1.0.0",
    description="Complete ML pipeline for energy consumption forecasting",
    author="Youssef AIT ELOURF",
    author_email="youssefaitelourf@gmail.com",
    url="https://github.com/yourusername/energy-forecasting-pipeline",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "xgboost>=2.0.0",
        "scipy>=1.11.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
