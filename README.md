### MExConn: A Mechanistically Interpretable Multi-Expert Framework for Multi-Organelle Segmentation in Connectomics
This repository contains  the official codebase for our work **MExConn** for multi-organelle segmentation from electron microscopy (EM) images with Mechanistic Interpretability.

## Dataset
The processed datasets are available at https://drive.google.com/file/d/1lUity8Gs4KmWSYWnO_sbZ71EFltuFAOP/view?usp=sharing

## Usage
1. Training MExConn :

```
python train_mexconn.py --domain drosophila-vnc train 
python train_mexconn.py --domain multiclass train   
python train_mexconn.py --domain urocell_3 train 
```
2. Testing MExConn :

```
python train_mexconn.py --domain drosophila-vnc test 
python train_mexconn.py --domain multiclass test   
python train_mexconn.py --domain urocell_3 test 
```
3. Mechanistic Interpretability Analysis :

Compute encoder channel importance:
```
python channel_ablation.py --data_root data --domain drosophila-vnc --model_path ./models/drosophila/model.pth --batch_size 8 --top_k 100 --device cuda

```
Detailed annotations are provided in the code for each step of the interpretability analysis.
