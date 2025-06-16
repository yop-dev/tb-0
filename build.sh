#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Configure Poetry to use system Python
poetry config virtualenvs.create false

# Install dependencies using Poetry
poetry install --no-interaction --no-ansi 