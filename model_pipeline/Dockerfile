# Use the official Debian-hosted Python image
FROM python:3.8-slim-buster

# Tell pipenv where the shell is. 
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

# Ensure we have an up to date baseline, install dependencies 
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    pipenv \
    pandas \
    torch \
    torchvision \
    tqdm \
    scikit-learn \
    pillow

# Add Pipfile, Pipfile.lock + python code
ADD Pipfile Pipfile.lock /app/

RUN pipenv sync

# Entry point
ENTRYPOINT ["/bin/bash"]

# Get into the pipenv shell
CMD ["-c", "pipenv shell"]