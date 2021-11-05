from setuptools import setup, find_packages
with open("README.md") as fh:
    long_description = fh.read()
setup(
    name='ivp_enhanced',
    version='0.0.2',
    packages=find_packages(include=['ivp_enhanced', 'ivp_enhanced.*']),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/zhaofeng-shu33/ivp-enhanced',
    author='zhaofeng-shu33',
    author_email='616545598@qq.com',
    description='enhance some missing features of scipy.integrate.solve_ivp'
)