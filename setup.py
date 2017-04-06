#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name='astrotools',
    version='2.0',
    description='Python Astro Tools',
    author='Gero Müller, Martin Urban, David Walz, Marcus Wirtz',
    author_email='gmueller@physik.rwth-aachen.de, murban@physik.rwth-aachen.de, walz@physik.rwth-aachen.de, '
                 'mwirtz@physik.rwth-aachen.de',
    url='https://forge.physik.rwth-aachen.de/projects/astrotools',
    packages=['astrotools'],
    package_data={'astrotools': ['data/*.txt', 'data/lnA/*', 'data/xmax/*']},
    requires=['numpy']
)
