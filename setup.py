#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
import numpy as np


tpc = Extension('astrotools._tpc', 
  sources=['astrotools/tpc.c'], 
  extra_compile_args=['-fopenmp', '-Ofast', '-g'],
  extra_link_args=['-lgomp']
  )

setup(
  name='astrotools',
  version='1.0',
  description='Python Astro Tools',
  author='Gero Müller, David Walz',
  author_email='gmueller@physik.rwth-aachen.de, walz@physik.rwth-aachen.de',
  url='https://forge.physik.rwth-aachen.de/projects/astro-tools',
  packages=['astrotools'],
  include_dirs=[np.get_include()],
  ext_modules=[tpc],
  package_data={'astrotools': ['*.txt']},
  )
