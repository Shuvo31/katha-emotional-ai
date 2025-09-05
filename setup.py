from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="katha-emotional-ai",
    version="0.1.0",
    author="Shuvojit Das",
    author_email="Shuvojitdas2000@gmail.com",
    description="Emotional AI Voice for Indian Languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shuvo31/katha-emotional-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",  # Updated for better Python 3.13 compatibility
        "transformers>=4.20.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "numpy>=1.24.0",  # Updated for Python 3.13
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",  # Updated for better performance
        "scipy>=1.10.0",
        "TTS>=0.13.0",
        "langdetect>=1.0.9",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "katha=src.katha_core:main",
            "katha-demo=demo.streamlit_app:main",
        ],
    },
    keywords=[
        "ai", "voice", "tts", "emotion", "indian-languages", 
        "bengali", "hindi", "nlp", "speech-synthesis"
    ],
    project_urls={
        "Bug Reports": "https://github.com/Shuvo31/katha-emotional-ai/issues",
        "Source": "https://github.com/Shuvo31/katha-emotional-ai",
        "Documentation": "https://github.com/Shuvo31/katha-emotional-ai/docs",
    },
)