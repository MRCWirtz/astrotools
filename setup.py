#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
import numpy as np

setup(
  name='astrotools',
  version='1.0',
  description='Python Astro Tools',
  author='Gero Müller, David Walz',
  author_email='gmueller@physik.rwth-aachen.de, walz@physik.rwth-aachen.de',
  url='https://forge.physik.rwth-aachen.de/projects/astro-tools',
  packages=['astrotools'],
  package_data={'astrotools': ['*.txt']},
  )
