from setuptools import setup, find_packages
import os
import json

with open("version.json", "r") as f:
    version = json.load(f)

version_name = '{major}.{build}.{minor}'.format(**version)

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if os.path.isfile(os.path.join(here, 'requirements.txt')):
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        pipreq = f.readlines()
        # remove pip flag
        if '-i http' in pipreq[0]:
            pipreq.pop(0)
else:
    pipreq = ['pyyaml']


setup(
    name="natural_selection",
    version=version_name,
    description='Tools for running evolutionary algorithm experiments',
    # long_description=long_description,
    license='Apache 2.0',
    keywords = ['GENETIC ALGORITHMS', 'EVOLUTIONARY ALGORITHMS'],
    author='Zipfian Science',
    author_email='about@zipfian.science',
    zip_safe=False,
    # url='https:/zipfian.science',
    download_url='https://github.com/Zipfian-Science/natural-selection/archive/v_01.tar.gz',
    packages=find_packages(".", exclude=["tests", "dist", "deploy", "egg-info"]),
    include_package_data=True,
    install_requires=pipreq,
    package_dir={'.': 'natural_selection'},
    package_data={
        "": ["*.yaml",],
    },
    classifiers=[
               # 'Intended Audience :: Researchers',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Topic :: Utilities']
)
