from setuptools import setup, find_packages

setup(
    name="benefitsbot",
    version="1.0",
    packages=find_packages(
        
    ),
    # install_requires=[
    #     # Dependencies will be installed from requirements.txt
    # ],
    python_requires=">=3.10",
    # Include non-Python files
    include_package_data=True,
    # Add any scripts that should be in PATH
    # scripts=[
    #     "analysis/benefitsbot.py",
    # ],
)
