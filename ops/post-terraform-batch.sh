#!/usr/bin/env bash

mkdir ~/workspace
cd ~/workspace

git clone git@bitbucket.org:thoolihan/dlbook.git
cd dlbook

conda env create -f environment.yml
source activate dlb

pip install kaggle
kaggle competitions download -c titanic -p data/titanic/
kaggle competitions download -c dogs-vs-cats -p data/dogs_cats/original/

cd data/dogs_cats/original/
unzip train
unzip test1
cd ../../../
python ex05_02_prep.py
