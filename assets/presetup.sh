#!/bin/bash

# Update system packages
sudo apt-get update

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
rm Miniconda3-py38_23.5.2-0-Linux-x86_64.sh

sudo apt-get update
source ~/.bashrc

# You should RESTART your TERMINAL


# # Create a new Conda environment
# conda create -n newenv python=3.8 anaconda
# conda activate newenv