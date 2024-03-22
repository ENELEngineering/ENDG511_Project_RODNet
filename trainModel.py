# Copyright 2024. All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential.
#
# This source code is provided solely for runtime interpretation by Python.
# Modifying or copying any source code is explicitly forbidden.
# 
# This python file is used explicitly to meet the project requirements provided
# in ENDG 511 at the University of Calgary.

from __future__ import annotations
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from getModel import RODNetBranched, RODNetBase
    from torch.utils.data import DataLoader

import torch
import json
import os


class TrainBranchedHandler():
    """
    This class handles training of branched models.
    """
    def __init__(
            self, 
            net: RODNetBranched, 
            criterion: Union[torch.nn.modules.loss.BCELoss], 
            optimizer: Union[torch.optim.adam.Adam], 
            device: torch.device, 
            scheduler: Union[torch.optim.lr_scheduler.StepLR]=None, 
            num_epochs: int=1,
            half: bool=False
        ):

        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.NUM_EPOCHS = num_epochs
        self.device = device
        self.half = half
        if half:
            self.scaler = torch.cuda.amp.GradScaler()

        self.history = {
            "short_branched": {
                "train": {"loss": []},
            },

            "long_branched": {
                "train": {"loss": []},
            },

            "total": {
                "train": {"loss": []},
            }
        }
        
    def train(
            self, 
            train_loader: DataLoader, 
            model_dir: str= ""
        ):

        for epoch in range(self.NUM_EPOCHS):
            total1loss, total2loss, totalloss = 0, 0, 0

            self.net.train()
            for iter, data_dict in enumerate(train_loader):
                print(f"{iter=}", end="\r")
                # Getting data
                data = data_dict['radar_data'].to(self.device)
                confmap_gt = data_dict['anno']['confmaps'].to(self.device)
                # Forward
                self.optimizer.zero_grad()
                if self.half:
                    # Enables autocasting for the forward pass (model + loss)
                    with torch.cuda.amp.autocast():
                        confmap_preds_1,confmap_preds_2 = self.net(data.float())
                        loss_confmap_1 = self.criterion(confmap_preds_1, confmap_gt.float())
                        loss_confmap_2 = self.criterion(confmap_preds_2, confmap_gt.float())
                else:
                    confmap_preds_1,confmap_preds_2 = self.net(data.float())
                    loss_confmap_1 = self.criterion(confmap_preds_1, confmap_gt.float())
                    loss_confmap_2 = self.criterion(confmap_preds_2, confmap_gt.float())
                # Backward
                total1loss += loss_confmap_1.item()
                total2loss += loss_confmap_2.item()
                totalloss += 0.5*loss_confmap_1.item() + 0.5*loss_confmap_2.item()
                if self.half:    
                    self.scaler.scale(loss_confmap_1).backward(
                        inputs=list(self.net.encoder.parameters()) + list(self.net.decoder_short.parameters()), 
                        retain_graph=True
                    )
                    self.scaler.scale(loss_confmap_2).backward(
                        inputs=list(self.net.decoder_long.parameters())
                    )
                else:
                    loss_confmap_1.backward(
                        inputs=list(self.net.encoder.parameters()) + list(self.net.decoder_short.parameters()), 
                        retain_graph=True)
                    loss_confmap_2.backward(
                        inputs=list(self.net.decoder_long.parameters()))
                self.optimizer.step()

            self.scheduler.step()
            total1loss = total1loss/len(train_loader)
            total2loss = total2loss/len(train_loader)
            totalloss = totalloss/len(train_loader)
            
            self.history["short_branched"]["train"]["loss"].append(total1loss)
            self.history["long_branched"]["train"]["loss"].append(total2loss)
            self.history["total"]["train"]["loss"].append(totalloss)
            
            print("epoch {} --> short branch loss: {:0.3f}, long branch loss: {:0.3f}, total loss: {:0.3f}"
              .format(epoch+1, total1loss, total2loss, totalloss))
            
            print("saving current epoch model ...")
            save_model_path = os.path.join(model_dir, "multi_epoch_%02d_final.pkl" % (epoch + 1))
            torch.save(self.net.state_dict(), save_model_path)

        return self.net, self.history
    
    def save_metrics(self, results_dir=""):
        save_model_path = os.path.join(results_dir, "training_metrics_branched.json")
        with open(save_model_path, 'w') as fp:
            json.dump(self.history, fp)
    

class TrainBaseHandler():
    """
    This class handles training of the base RODNet model.
    """
    def __init__(
            self, 
            net: RODNetBase, 
            criterion: Union[torch.nn.modules.loss.BCELoss], 
            optimizer: Union[torch.optim.adam.Adam], 
            device: torch.device, 
            scheduler: Union[torch.optim.lr_scheduler.StepLR]=None, 
            num_epochs: int=1,
            half: bool=False
        ):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.NUM_EPOCHS = num_epochs
        self.device = device
        self.half = half
        if half:
            self.scaler = torch.cuda.amp.GradScaler()

        self.history = {
            "train": {"loss": []}
        }
        
    def train(
            self, 
            train_loader: DataLoader, 
            model_dir: str= ""
        ):

        for epoch in range(self.NUM_EPOCHS):
            totalloss = 0

            self.net.train()
            for iter, data_dict in enumerate(train_loader):
                print(f"{iter=}", end="\r")
                # Getting data
                data = data_dict['radar_data'].to(self.device)
                confmap_gt = data_dict['anno']['confmaps'].to(self.device)
                # Forward
                self.optimizer.zero_grad()
                if self.half:
                    # Enables autocasting for the forward pass (model + loss)
                    with torch.cuda.amp.autocast():
                        confmap_preds = self.net(data.float())
                        loss_confmap = self.criterion(confmap_preds, confmap_gt.float())
                else:
                    confmap_preds = self.net(data.float())
                    loss_confmap = self.criterion(confmap_preds, confmap_gt.float())
                
                # Backward
                totalloss += loss_confmap.item()
                if self.half:    
                    self.scaler.scale(loss_confmap).backward()
                    # self.scaler.step(self.optimizer) # NOSONAR => Issue with FP16
                    # self.scaler.update()
                else:
                    loss_confmap.backward()
                self.optimizer.step()
                
            self.scheduler.step()
            totalloss = totalloss/len(train_loader)
            self.history["train"]["loss"].append(totalloss)

            print("epoch {} --> trainLoss: {:0.3f}".format(epoch+1, totalloss))
            print("saving current epoch model ...")
            save_model_path = os.path.join(model_dir, "base_epoch_%02d_final.pkl" % (epoch + 1))
            torch.save(self.net.state_dict(), save_model_path)

        return self.net, self.history

    def save_metrics(self, results_dir=""):
        save_model_path = os.path.join(results_dir, "training_metrics_base.json")
        with open(save_model_path, 'w') as fp:
            json.dump(self.history, fp)