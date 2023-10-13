from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="viral-conversation",
    version="0.1",
    description="ChatGPT conversations on social media platforms and its comparison with the COVID-19 topic",
    author="Shayan Alipour",
    author_email="shayan.alipour@uniroma1.it",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
