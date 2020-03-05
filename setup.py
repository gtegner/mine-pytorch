import setuptools

setuptools.setup(
    name="mine", # Replace with your own username
    version="0.0.1",
    author="",
    author_email="",
    description="An implementation of the MINE algorithm in Pytorch",
    long_description="",
    long_description_content_type="",
    url="https://github.com/gtegner/mine-pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        'pytorch_lightning==0.6.0',
        'matplotlib==3.1.2',
        'numpy==1.17.4',
        'tqdm==4.40.2',
        'torch==1.3.1',
        'torchvision==0.4.2',
        'scikit_learn==0.22.1',
        'pillow<7'
    ]
)