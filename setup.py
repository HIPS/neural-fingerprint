from __future__ import absolute_import
from distutils.core import setup

setup(name='neuralfingerprint',
      version='1.0.0',
      description='Computes differentiable fingerprints of molecular graphs.',
      author='David Duvenaud and Dougal Maclaurin',
      author_email="dduvenaud@seas.harvard.edu, maclaurin@physics.harvard.edu",
      packages=['neuralfingerprint'],
      install_requires=['numpy>=1.8', 'scipy>=0.15', 'autograd'],
      keywords=['Chemistry', 'Molecular Fingerprints', 'Morgan fingerprints',
                'machine learning', 'Circular fingerprints', 'neural networks',
                'Python', 'Numpy', 'Scipy'],
      url='https://github.com/HIPS/neural-fingerprint',
      license='MIT',
      classifiers=['Development Status :: 4 - Beta',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 2.7'])
