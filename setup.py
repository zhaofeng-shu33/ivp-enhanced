from setuptools import setup, find_packages
with open("README.md") as fh:
    long_description = fh.read()
setup(
    name='ivp_enhanced',
    version='0.0.1',
    long_description = long_description,
    long_description_content_type="text/markdown"
)