# mri-features

This repository contains code used to extract features from MRI images in the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. The features are generated by training a CNN to classify images using the features as an intermediate representation. They are then fed to the FLARe model upstream, where they are used to make forecasting predictions about a patient's clinical status.

## Instructions

Log onto GYPSUM and run the following commands.

```
git clone git@github.com:Information-Fusion-Lab-Umass/mri-features.git
cd mri-features
cp /path/to/mriimg_meta_v4.csv .

conda create -n liuNetEnv python=3.6
conda activate liuNetEnv
pip install -r requirements.txt
```

Once you've finished setting up, you can begin the actual training process.

```
# For each N from 0 to 9...

screen -S splitN
# Allocate GPU resources for our program on a new node
srun --pty --partition=titanx-long --gres=gpu:1 /bin/bash
conda activate liuNetEnv
# Generate features for the Nth split
python run_liunet.py --split N

# Use Ctrl+A D to detach from the screen for this split
```