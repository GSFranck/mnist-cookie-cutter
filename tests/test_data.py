from tests import _PATH_DATA
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models.model import ConvNet
import os.path
import pytest


dataset = _PATH_DATA+'/processed'
images = torch.unsqueeze( torch.load(f'{dataset}/train_images.pt'), dim=1)
labels = torch.load(f'{dataset}/train_labels.pt')
train = TensorDataset(images,labels)
train_set = DataLoader(train, batch_size=8, shuffle=True)

@pytest.mark.skipif(not os.path.exists(dataset), reason="Data files not found")

def test_data_length():
    assert len(images)==25000

def test_data_shape():
    for image in images:
        assert image.shape==(1, 28, 28)
def test_label_length():
    assert len(labels)==25000

def test_label_rep():
    unique_labels=torch.unique(labels)
    for i in range (10):
        assert i in unique_labels