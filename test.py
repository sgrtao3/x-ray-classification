import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn as nn

from models import RestNet18, X_ray_Classifier
from utils import train, validate, set_seed, preds_average # Noqa
from datasets import AugmentedBalancedDataset, AugmentedDataset


class SimpleNet(nn.Module):
    """
    This model is here as a fixture that the other test functions
    can use as a very simple model for the tests.
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear_1 = nn.Linear(28*28, 25)
        self.linear_2 = nn.Linear(25, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        z1 = self.linear_1(x)
        a1 = self.activation(z1)
        z2 = self.linear_2(a1)
        return z2


def test_train():
    """
    Tests the train function. Very simple test to just make sure that
    the output types of the train function are correct.
    """
    set_seed(42) # Noqa
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_train = MNIST("./", download=True, train=False, transform=transform)
    model = SimpleNet()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True,
                              num_workers=0)
    loss, accuracy = train(model, optimizer, criterion, train_loader,
                           imshape=(-1, 28*28))

    assert type(loss) == torch.Tensor
    assert type(accuracy) == np.float64
    assert len(loss.shape) == 0


def test_preds_average():
    """
    Tests whether the function to average mutliple models' predictions works
    """
    pred_1 = np.array([[0.1, 0.3, 0.1, 0.5], [0.9, 0.05, 0.025, 0.025]])
    pred_2 = np.array([[0.6, 0.1, 0.2, 0.1], [0.8, 0.1, 0.05, 0.05]])
    av = preds_average([pred_1, pred_2], [0.9, 0.1])
    assert (av == np.array([3, 0])).all()


def test_validate():
    """
    Tests the validation function. Very simple test to just make sure that
    the output types of the validation function are correct.
    """
    set_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist_train = MNIST("./", download=True, train=False, transform=transform)
    model = SimpleNet()

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True,
                              num_workers=0)
    loss, accuracy = validate(model, criterion, train_loader,
                              imshape=(-1, 28*28))

    assert type(loss) == torch.Tensor
    assert type(accuracy) == np.float64
    assert len(loss.shape) == 0


def test_resnet18():
    """
    Tests the initialization of the implemented Resnet 18 model
    """
    model = RestNet18()
    assert type(model) == RestNet18


def test_xray_classifier():
    """
    Tests the initialization of the custom X-Ray classifier model
    """
    model = X_ray_Classifier()
    assert type(model) == X_ray_Classifier


def test_dataaug():
    """
    Function to test the functionality of the data augmentation datasets
    """
    # test len
    augdataset = AugmentedDataset(4, foldername="test_data")
    assert len(augdataset) == 40

    augbaldatset = AugmentedBalancedDataset(2, foldername="test_data")
    assert len(augbaldatset) == 2*(3*4)

    # test getitem
    assert augdataset[35][0].size() == (1, 299, 299)
    assert augbaldatset[40040][0].size() == (1, 299, 299)
