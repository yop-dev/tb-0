#!/usr/bin/env bash
# exit on error
set -o errexit

# Install setuptools first
pip install --upgrade pip setuptools wheel

# Then install the rest of the requirements
pip install -r requirements.txt 