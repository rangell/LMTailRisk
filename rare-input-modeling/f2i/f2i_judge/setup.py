from setuptools import setup, find_packages

setup(
    name="crimson-score",
    version="0.1.0",
    description="CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation",
    author="Mohammed Baharoon, Thibault Heintz, Siavash Raissi, Mahmoud Alabbad, Mona Alhammad, Hassan AlOmaish, Sung Eun Kim, Oishi Banerjee, Pranav Rajpurkar",
    license="MIT",
    packages=find_packages(include=["CRIMSON*"]),
    python_requires=">=3.8",
    install_requires=[
        # "openai>=2.26.0",
        # "transformers<=4.57.3",
        # "torch==2.8.0",
        # # "peft>=0.18.1",
        # # "accelerate>=1.13.0",
        # # "pandas>=3.0.1",
        # # "numpy>=2.4.3",
        # # "scipy>=1.17.1",
        # "tqdm>=4.67.3",
        # # "pyyaml>=6.0.3",
        # # "pydantic>=2.12.5",
        # "sentence-transformers>=5.2.3",
        "huggingface_hub==0.34.0",
        # "scikit-learn>=1.8.0",
        # "openpyxl>=3.1.5",
        # "Pillow>=12.1.1",
    ],
    extras_require={
        "dev": ["pytest", "twine"],
    },
    url="https://github.com/rajpurkarlab/CRIMSON",
    keywords=["radiology", "report evaluation", "medical AI", "NLP", "LLM"],
)
