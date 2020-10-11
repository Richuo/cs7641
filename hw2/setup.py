""" MLROSe setup file."""

# Author: Genevieve Hayes
# Modified: Andrew Rollings
# License: BSD 3 clause

from setuptools import setup


def readme():
    """
    Function to read the long description for the MLROSe package.
    """
    with open('README.md') as _file:
        return _file.read()


setup(name='mlrose_hiive2',
      version='2.1.3',
      description="MLROSe: Machine Learning, Randomized Optimization and"
      + " Search (hiive extended remix)",
      url='https://github.com/hiive/mlrose',
      author='Genevieve Hayes (modified by Andrew Rollings)',
      license='BSD',
      download_url='https://github.com/hiive/mlrose/archive/2.1.3.tar.gz',
      classifiers=[
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English",
          "License :: OSI Approved :: BSD License",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      packages=['mlrose_hiive2','mlrose_hiive2.runners','mlrose_hiive2.generators', 'mlrose_hiive2.algorithms',
                'mlrose_hiive2.algorithms.decay', 'mlrose_hiive2.algorithms.crossovers',
                'mlrose_hiive2.opt_probs', 'mlrose_hiive2.fitness', 'mlrose_hiive2.algorithms.mutators',
                'mlrose_hiive2.neural', 'mlrose_hiive2.neural.activation', 'mlrose_hiive2.neural.fitness',
                'mlrose_hiive2.neural.utils', 'mlrose_hiive2.decorators',
                'mlrose_hiive2.gridsearch'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'networkx', 'joblib'],
      python_requires='>=3',
      zip_safe=False)
