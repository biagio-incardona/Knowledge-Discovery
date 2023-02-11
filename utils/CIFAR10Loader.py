from torch.utils.data import DataLoader
from utils.dataset_utils import TrainingDatasetFF, DatasetLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, Normalize
import torch
class CIFAR10Loader(DatasetLoader):
    """MNIST PyTorch Dataset Loader
    """

    def __init__(self, download_path: str = './tmp'):
        """Initialize MNIST Dataset Loader

        Args:
            train_transform (torchvision.transforms): transformations to be applied to training set
            test_transform (torchvision.transforms): transformations to be applied to test set
            download_path (str, optional): download path. Defaults to './tmp'.
        """
        super(CIFAR10Loader, self).__init__()
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        Lambda(lambda x: torch.flatten(x))])
        self.train_transform = transform
        self.test_transform = transform

        self.download_path = download_path

    def download_dataset(self) -> None:
        """Download dataset to the given path
        """
        self.train_set = torchvision.datasets.CIFAR10(root=self.download_path, train=True,
                               download=True,
                               transform=self.train_transform)

        self.test_set = torchvision.datasets.CIFAR10(self.download_path, train=False,
                              download=True,
                              transform=self.test_transform)