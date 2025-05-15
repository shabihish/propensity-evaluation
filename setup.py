from setuptools import setup, find_packages

setup(
    name="my_simple_agent_package",  # You can change this to your desired package name
    version="0.1.0",                # A starting version
    packages=find_packages(),       # This will automatically find the 'agent' package
)