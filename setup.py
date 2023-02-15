from setuptools import setup, find_packages
import os
from datetime import datetime

here = os.path.abspath(os.path.dirname(__file__))

about = {}

with open(os.path.join(here, "natural_selection", "__version__.py")) as f:
    exec(f.read(), about)

# Get the long description from the README file
long_description_file = os.path.join(here, 'pip_description.md')
if os.path.isfile(long_description_file):
    with open(long_description_file, "r", encoding='utf-8') as f:
        long_description = f.read()
        long_description = long_description.format(pypi_metdata_release_date=datetime.today().strftime('%Y-%m-%d'),
                                                   pypi_metdata_version_number=about["__version__"])
else:
    long_description = "EA for Python!"

if os.path.isfile(os.path.join(here, 'requirements.txt')):
    with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
        pipreq = f.readlines()
        # remove pip flag
        if '-i http' in pipreq[0]:
            pipreq.pop(0)
else:
    pipreq = ['numpy']


setup(
    name="natural_selection",
    version=about["__version__"],
    description='Tools for running evolutionary algorithm experiments',
    long_description_content_type='text/markdown',
    long_description=long_description,
    license='Apache 2.0',
    keywords = ['GENETIC ALGORITHMS', 'EVOLUTIONARY ALGORITHMS'],
    author='Zipfian Science',
    author_email='about@zipfian.science',
    zip_safe=False,
    url='https://github.com/Zipfian-Science/natural-selection',
    download_url='https://github.com/Zipfian-Science/natural-selection/archive/v_01.tar.gz',
    packages=find_packages(".", exclude=("tests", "dist", "deploy", "egg-info")),
    include_package_data=True,
    install_requires=pipreq,
    package_dir={'.': 'natural_selection'},
    classifiers=[
            'Intended Audience :: Science/Research',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence']
)
