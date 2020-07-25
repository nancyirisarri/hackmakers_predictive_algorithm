from setuptools import find_packages, setup

setup(
    name='hackmakers_predictive_algorithm',
    packages=find_packages(),
    version='0.0.1',
    description='Repository for the Hackmakers hackathon predictive algorithm challenge.',
    author='Nancy Irisarri',
    license='',
    long_description="README.md",

    python_requires='>3.5',

    install_requires=[
                     "click",
                     "python-dotenv>=0.5.1",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
