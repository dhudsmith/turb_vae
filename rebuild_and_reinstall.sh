#!/usr/bin/bash

python setup.py sdist bdist_wheel

conda activate turb
pip uninstall -y turb_vae
pip install .