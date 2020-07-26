from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='fl-baseline-data',
      version='0.1',
      description='Baseline data for federated learning',
      url='http://github.com/amitport/fl-baseline-data',
      author='Amit Portnoy',
      author_email='amit.portnoy@gmail.com',
      packages=['fl_baseline_data'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      zip_safe=True,
      install_requires=[
          'tensorflow>=2',
          'tensorflow-datasets>=3'
      ],
      python_requires='>=3.6',
      )
