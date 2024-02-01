#!/bin/bash

# # Specify the Conda environment name
# env_name="newenv"

# # Specify the desired packages to be installed
# #packages=("numpy" "matplotlib" "pandas")

# # Create the Conda environment
# #conda create -y -n "$env_name" "${packages[@]}"
# conda create -y -n "$env_name" python=3.8 anaconda

# # Activate the newly created environment
# conda activate "$env_name"

# # Install additional packages if needed
# # conda install -y package_name

# # Display a message
# echo "Conda environment '$env_name' has been created and activated."

# # Activate the Conda environment
# source activate $conda_env_name

# Install packages or run commands in the activated environment
# For example, let's install a package using pip
#conda install -y pip

# Install CUDA toolkit and other packages
conda install -c conda-forge cudatoolkit=11.0
sudo apt update
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install opencv-python
pip install scikit-learn
pip install gdown==4.6.0
pip install pillow==9.5.0

# Install necessary system packages
sudo apt-get update
sudo apt-get install unzip
sudo apt-get install libgl1-mesa-glx

# Update system packages again
sudo apt-get update

# Add Ubuntu Toolchain PPA and upgrade libstdc++
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get upgrade libstdc++6

# Install numpy version < 1.24
pip install "numpy<1.24"

# Install nano text editor
sudo apt-get install nano

# # Clone the repository
# git clone -b votdataset https://github.com/aerenkaradag/DeFRCN
# cd DeFRCN

# # Data Preparation
# gdown --fuzzy https://drive.google.com/file/d/1BcuJ9j9Mtymp56qGSOfYxlXN4uEVyxFm/view?usp=sharing -O VOC2007.zip
# unzip VOC2007.zip -d datasets/
# rm VOC2007.zip

# gdown --fuzzy https://drive.google.com/file/d/1NjztPltqm-Z-pG94a6PiPVP4BgD8Sz1H/view?usp=sharing -O VOC2012.zip
# unzip VOC2012.zip -d datasets/
# rm VOC2012.zip

# gdown --fuzzy https://drive.google.com/file/d/1BpDDqJ0p-fQAFN_pthn2gqiK5nWGJ-1a/view?usp=sharing -O vocsplit.zip
# unzip vocsplit.zip -d datasets/
# rm vocsplit.zip

gdown --fuzzy https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing -O ImageNetPretrained.zip
unzip ImageNetPretrained.zip
rm ImageNetPretrained.zip

gdown --fuzzy https://drive.google.com/file/d/1Ff5jP4PCDDPQ7lzsageZsauFWer73QIl/view?usp=sharing -O voc.zip
unzip voc.zip -d checkpoints/
rm voc.zip

cd checkpoints/voc/defrcn_one/
cp -r defrcn_fsod_r101_novel3 defrcn_fsod_r101_novel4
cp -r defrcn_gfsod_r101_novel3 defrcn_gfsod_r101_novel4
cp -r defrcn_det_r101_base3 defrcn_det_r101_base4



