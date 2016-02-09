from setuptools import setup, find_packages

setup(
    name='bigartm',
    version='0.7.4',
    packages=find_packages(),
    install_requires=[
        'pandas >= 0.16.2',
        'numpy >= 1.9.2',
    ],
)
