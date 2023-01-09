from src.models.model import ConvNet
from tests import _PATH_DATA
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytest
model=ConvNet()


dataset = _PATH_DATA+'/processed/'
images = torch.unsqueeze( torch.load(f'{dataset}/train_images.pt'), dim=1)
labels = torch.load(f'{dataset}/train_labels.pt')
train = TensorDataset(images,labels)
train_set = DataLoader(train, batch_size=8, shuffle=True)

def test_in_out():
    for images, labels in train_set:
        assert images.shape==(8,1,28,28)
        outputs = model(images)
        assert outputs.shape==(8,10)




def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))

def test_error_on_image_shape():
    with pytest.raises(ValueError, match='Expected each sample to have shape \[1, 28, 28\]'):
        model(torch.rand((25000,2,28,28)))
