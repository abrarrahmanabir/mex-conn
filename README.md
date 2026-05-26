### MExConn: A Mechanistically Interpretable Multi-Expert Framework for Multi-Organelle Segmentation in Connectomics
This repository contains  the official codebase for our work **MExConn** for multi-organelle segmentation from electron microscopy (EM) images with Mechanistic Interpretability.

# Dataset
The processed datasets are available at https://drive.google.com/file/d/1lUity8Gs4KmWSYWnO_sbZ71EFltuFAOP/view?usp=sharing

## Usage
1. Training MExConn :

```
python train_mexconn.py --domain drosophila-vnc train 
python train_mexconn.py --domain multiclass train   
python train_mexconn.py --domain urocell_3 train 

```
   
