from setuptools import setup

setup(
    name='bigartm',
    version='${PACKAGE_VERSION}', # FIXME: fill version variable in CMakeLists
    package_dir={'': '${CMAKE_CURRENT_SOURCE_DIR}'},
    packages=['artm'],
    install_requires=[
        'pandas',
    ],
)
