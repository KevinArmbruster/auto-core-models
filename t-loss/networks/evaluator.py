from typing import Any, Dict, Union, Optional, List, Type

import nni
from nni.nas.evaluator.pytorch import SupervisedLearningModule, Lightning, Trainer#, DataLoader

import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from losses.triplet_loss import TripletLoss, TripletLossVaryingLength


@nni.trace
class Causal_CNN_Classification(Lightning):
    def __init__(self, 
                 criterion: Type[nn.Module] = TripletLoss,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.,
                 optimizer: Type[optim.Optimizer] = optim.Adam,
                 train_dataloaders: Optional[DataLoader] = None,
                 train_set: Dataset = None,
                 **trainer_kwargs):
        
        module = SupervisedLearningModule(criterion=criterion, 
                                          learning_rate=learning_rate, 
                                          weight_decay=weight_decay, 
                                          optimizer=optimizer, 
                                          export_onnx=None)
        
        super().__init__(module, 
                         Trainer(**trainer_kwargs),
                         train_dataloaders=train_dataloaders,
                         val_dataloaders=None,
                         datamodule=None)
        
        self.metrics = []
        self.train_dataloaders = train_dataloaders
        self.train_set = train_set
    
    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        loss = self.criterion(batch, self, self.train_set, save_memory=False)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    