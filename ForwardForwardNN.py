from pathlib import Path
from tqdm.notebook import tqdm
from typing import Callable, Tuple
import torch
from torch.nn import Linear, Module
from utils.dataset_utils import MNISTLoader, TrainingDatasetFF
from utils.models import FFSequentialModel
from utils.tools import base_loss, generate_positive_negative_samples_overlay
from utils.layers import FFLinear
from torchvision.transforms import Compose, ToTensor, Lambda, Normalize
import warnings

class FowardForwardNN(Module, FFSequentialModel):
    
    def __init__(self, train_batch_size: int = 1024, test_batch_size: int = 1024, layers: torch.nn.ModuleList[FFLinear] = []) -> None:
        super(FowardForwardNN, self).__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_train_batch_size(train_batch_size)
        self.set_test_batch_size(test_batch_size)
        self._layers = layers
        self._network = None

    def set_train_batch_size(self, train_batch_size: int = 1024) -> object:
        self._train_batch_size = train_batch_size
        return self

    def set_test_batch_size(self, test_batch_size: int = 1024) -> object:
        self._test_batch_size = test_batch_size
        return self

    def get_train_batch_size(self) -> int:
        return self._train_batch_size
    
    def get_test_batch_size(self) -> int:
        return self._test_batch_size
    
    def add_layer(self, layer: FFLinear, position: int = -1) -> object:
        if position == -1:
            self._layers.append(layer)
        elif position > len(self._layers):
            warnings.warn(f"position {position} is higher than the list length. Layer added at the end of the network")
            self._layers.append(layer)
        elif position <=0:
            warnings.warn(f"position {position} is non positive and different than -1. Layer added at the beginning of the network")
            self._layers.insert(0, layer)
        else:
            self._layers.insert(position-1, layer)
        return self

    def drop_layer(self, position: int) -> object:
        try:
            self._layers.pop(position-1)
        except:
            raise ValueError(f"Position of layer not valid, please use value between 1 and {len(self._layers)}")
        return self
            
    def get_layers(self) -> list[FFLinear]:
        return self._layers
    
    def save(self) -> object:
        return self

    def fit() -> None:
        pass
