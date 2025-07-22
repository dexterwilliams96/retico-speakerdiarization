#!/usr/bin/env python3

"""
Setup script.

Use this script to install a speech diarization module for retico. Usage:
    $ python3 setup.py install

Author: Dexter Williams (dexterwilliams4355@gmail.com)
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

exec(open("retico_speakerdiarization/version.py").read())

import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

config = {
    "description": "Speech Diarization for retico",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author": "Dexter Williams",
    "author_email": "dexterwilliams4355@gmail.com",
    "url": "https://github.com/dexterwilliams96/retico-speakerdiarization",
    "download_url": "https://github.com/dexterwilliams96/retico-speakerdiarization",
    "version": __version__,
    "python_requires": ">=3.9, <4",
    "keywords": "retico, framework, incremental, dialogue, dialog",
    "install_requires": [
        "speechbrain @ git+https://github.com/speechbrain/speechbrain.git@develop",
        "retico-core>=0.2.10,<0.3.0",
        "torch>=2.7.1,<3.0.0",
        "torchaudio>=2.7.1,<3.0.0",
        "pydub>=0.25.1,<0.26.0",
        "webrtcvad>=2.0.10,<3.0.0",
        "sortedcontainers>=2.4.0,<3.0.0",
        ],
    "packages": find_packages(),
    "name": "retico-speakerdiarization",
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
}

setup(**config)
