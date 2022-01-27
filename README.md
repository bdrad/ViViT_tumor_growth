Implementing ViViT with 3DMNIST dataset 

ViViT_toy_dataset.ipynb: code with model implementation 

iter30_vivit_toy.npz: saved data for 30 iterations training loop in ViViT_toy_dataset.ipynb
MNIST_3D_data.py: classes to process and generate growing and shrinking 3D MNIST data


module.py: utils for vivit.py; taken from https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
vivit.py: ViViT model from https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
    - modified a bit to take in 3D + time data
