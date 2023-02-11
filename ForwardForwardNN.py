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
from torch.utils.data import DataLoader
from numpy import nan
import itertools


class FowardForwardNN(Module, FFSequentialModel):

    def __init__(self, train_batch_size: int = 1024, test_batch_size: int = 1024, layers: torch.nn.ModuleList = None) -> None:
#        super(FowardForwardNN, self).__init__()
        print(layers)
        self._device = 'cuda'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_train_batch_size(train_batch_size)
        self.set_test_batch_size(test_batch_size)
        self._layers = [] if layers is None else layers
        self._network = None
        self._last_train_accuracy = nan
        self._last_test_accuracy = nan
        self._n_classes = -1
        self._train_loader_ff = None


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
            
    def get_layers(self) -> list:
        return self._layers
    
    def _check_classes(self, train_loader: DataLoader, n_classes: int = None) -> None:
        if n_classes is None:
            if self._n_classes == -1:
                self._n_classes = len(set(list(itertools.chain([Y.tolist() for X, Y in train_loader]))[0]))
        elif self._n_classes == -1:
            self._n_classes = n_classes

    def fit(self, train_loader: DataLoader, before: bool, n_epochs: int = 10, n_classes: int = None, reload: bool = False) -> None:
        self._check_classes(train_loader, n_classes)
        if self._train_loader_ff is None or reload == True:
            self._train_loader_ff = torch.utils.data.DataLoader(TrainingDatasetFF(generate_positive_negative_samples_overlay(X.to(self._device),
                                                                           Y.to(self._device), False)
                                                                for X, Y in train_loader),
                                              batch_size=train_loader.batch_size, shuffle=True
                                              )
            
        for epoch in tqdm(range(n_epochs)):
            for X_pos, Y_neg in self._train_loader_ff:
                layer_losses = super(FowardForwardNN, self).train_batch(X_pos, Y_neg, before)
                print(", ".join(map(lambda i, l: 'Layer {}: {}'.format(i, l),list(range(len(layer_losses))) ,layer_losses)), end='\r')

    def test(self, loader: DataLoader, train: bool = False):
        acc = 0
        units = 0
        for X_, Y_ in tqdm(loader, total=len(loader)):
            X_ = X_.to(self._device)
            Y_ = Y_.to(self._device)
            units += len(Y_)
            acc += (self.predict_accomulate_goodness(X_,
                    generate_positive_negative_samples_overlay, n_class=self._n_classes).eq(Y_).sum())
        #df = loader.train_set if train == True else loader.test_set
        accuracy = acc/float(units)
        print(f"Accuracy: {accuracy:.4%}")
        print(f"{'Train' if train == True else 'Test'} error: {1 - accuracy:.4%}")

        if train == True:
            self._set_last_train_accuracy(accuracy)
        else:
            self._set_last_test_accuracy(accuracy)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.predict_accomulate_goodness(
            X,
            pos_gen_fn=generate_positive_negative_samples_overlay,
            n_class=self._n_classes
        )

    def _set_last_test_accuracy(self, accuracy: float) -> None:
        self._last_test_accuracy = accuracy

    def _set_last_train_accuracy(self, accuracy: float) -> None:
        self._last_train_accuracy = accuracy

    def get_last_train_accuracy(self) -> float:
        return self._last_train_accuracy
    
    def get_last_test_accuracy(self) -> float:
        return self._last_test_accuracy