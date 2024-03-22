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
    from cruw import CRUW

from cruw.eval.rod.load_txt import read_gt_txt, read_rodnet_res
from cruw.eval.rod.rod_eval_utils import compute_ols_dts_gts
from rodnet.core.post_processing import (
    write_dets_results_single_frame, 
    post_process_single_frame, 
)
import torch
import json
import os


class ValidateBranchedHandler():
    """
    Validate Branch Models
    """
    def __init__(
            self, 
            net: RODNetBase, 
            dataset: CRUW,
            config_dict: dict,
            criterion: Union[torch.nn.modules.loss.BCELoss], 
            device: torch.device
        ):
        self.net = net
        self.dataset = dataset
        self.config_dict = config_dict
        self.criterion = criterion
        self.device = device

        self.metrics = {
            "validation": {
                "short": {
                    "class_0": {
                        "loss": []
                    },
                    "class_1": {
                        "loss": []
                    },
                    "class_2": {
                        "loss": []
                    }
                },
                "long": {
                    "class_0": {
                        "loss": []
                    },
                    "class_1": {
                        "loss": []
                    },
                    "class_2": {
                        "loss": []
                    }
                },
                "ols_1": [],
                "ols_2": [],
                "early_exit_count": 0
            }
        }

    def validate(
            self,
            validation_loader: DataLoader,
            validation_dir: str=""
        ):
        
        with torch.no_grad():
            for iter, data_dict in enumerate(validation_loader):
                print(f"{iter=}")
                # Get Data.
                data = data_dict['radar_data'].to(self.device)
                confmap_gt = data_dict['anno']['confmaps'].to(self.device)
                seq_name = data_dict['seq_names'][0]
                start_frame = data_dict['start_frame'].item()
                # Get Model Inference.
                confmap_preds_1,confmap_preds_2 = self.net(data.float())
                # Calculate Loss.
                loss_1_class_0 = self.criterion(confmap_preds_1[:,0,:,:,:], 
                                                confmap_gt[:,0,:,:,:].float()) 
                loss_2_class_0 = self.criterion(confmap_preds_2[:,0,:,:,:], 
                                                confmap_gt[:,0,:,:,:].float())
                loss_1_class_1 = self.criterion(confmap_preds_1[:,1,:,:,:], 
                                                confmap_gt[:,1,:,:,:].float())
                loss_2_class_1 = self.criterion(confmap_preds_2[:,1,:,:,:], 
                                                confmap_gt[:,1,:,:,:].float())
                loss_1_class_2 = self.criterion(confmap_preds_1[:,2,:,:,:], 
                                                confmap_gt[:,2,:,:,:].float())
                loss_2_class_2 = self.criterion(confmap_preds_2[:,2,:,:,:], 
                                                confmap_gt[:,2,:,:,:].float())

                self.metrics["validation"]["short"]["class_0"]["loss"].append(loss_1_class_0)
                self.metrics["validation"]["short"]["class_1"]["loss"].append(loss_1_class_1)
                self.metrics["validation"]["short"]["class_2"]["loss"].append(loss_1_class_2)
                self.metrics["validation"]["long"]["class_0"]["loss"].append(loss_2_class_0)
                self.metrics["validation"]["long"]["class_1"]["loss"].append(loss_2_class_1)
                self.metrics["validation"]["long"]["class_2"]["loss"].append(loss_2_class_2)

                pred_1 = confmap_preds_1[0,:,0,:,:].cpu().detach().numpy()
                pred_2 = confmap_preds_2[0,:,0,:,:].cpu().detach().numpy()

                result_1 = post_process_single_frame(
                    pred_1, self.dataset, self.config_dict)
                result_2 = post_process_single_frame(
                    pred_2, self.dataset, self.config_dict)

                seq_res_dir = os.path.join(validation_dir, seq_name)
                if not os.path.exists(seq_res_dir):
                    os.makedirs(seq_res_dir)
                f1 = open(os.path.join(seq_res_dir, 'rod_res_1.txt'), 'a')
                f2 = open(os.path.join(seq_res_dir, 'rod_res_2.txt'), 'a')
                f1.close()
                f2.close()

                save_path_1 = os.path.join(
                    validation_dir, seq_name, 'rod_res_1.txt')
                save_path_2 = os.path.join(
                    validation_dir, seq_name, 'rod_res_2.txt')

                write_dets_results_single_frame(
                    result_1, start_frame, save_path_1, self.dataset)
                write_dets_results_single_frame(
                    result_2, start_frame, save_path_2, self.dataset)

                gt_path = os.path.join(
                    self.config_dict["dataset_cfg"]["anno_root"], 
                    f"train/{seq_name.upper()}.txt")
                data_path = os.path.join(
                    self.dataset.data_root, 
                    'sequences', 'train',
                    gt_path.split('/')[-1][:-4])
                n_frame = len(os.listdir(os.path.join(
                        data_path, 
                        self.dataset.sensor_cfg.camera_cfg['image_folder'])))
                gt_dets = read_gt_txt(gt_path, n_frame, self.dataset)
                sub_dets_1 = read_rodnet_res(save_path_1, n_frame, self.dataset)
                sub_dets_2 = read_rodnet_res(save_path_2, n_frame, self.dataset)
                # Continue if there are no detections.
                if None in [sub_dets_1, sub_dets_2]:
                    continue
                olss_all_1 = {(start_frame, catId): compute_ols_dts_gts(
                    gt_dets, sub_dets_1, start_frame, catId, self.dataset) \
                    for catId in range(3)}
                olss_all_2 = {(start_frame, catId): compute_ols_dts_gts(
                    gt_dets, sub_dets_2, start_frame, catId, self.dataset) \
                    for catId in range(3)}
                self.metrics["validation"]["ols_1"].append(olss_all_1)
                self.metrics["validation"]["ols_2"].append(olss_all_2)

                if(len(gt_dets[start_frame,0]) == len(sub_dets_1[start_frame,0]) and 
                   len(gt_dets[start_frame,1]) == len(sub_dets_1[start_frame,1]) and 
                   len(gt_dets[start_frame,2]) == len(sub_dets_1[start_frame,2])):
                    self.metrics["validation"]["early_exit_count"] += 1
        
    def save_metrics(self, results_dir: str=""):
        save_model_path = os.path.join(results_dir, "validation_metrics_branch.json")
        with open(save_model_path, 'w') as fp:
            json.dump(self.metrics, fp)


class ValidateBaseHandler():
    """
    Validate Base Models
    """
    def __init__(
            self, 
            net: RODNetBase, 
            dataset: CRUW,
            config_dict: dict,
            criterion: Union[torch.nn.modules.loss.BCELoss], 
            device: torch.device
        ):
        self.net = net
        self.dataset = dataset
        self.config_dict = config_dict
        self.criterion = criterion
        self.device = device

        self.metrics = {
            "validation": {
                "class_0": {
                    "loss": []
                },
                "class_1": {
                    "loss": []
                },
                "class_2": {
                    "loss": []
                },
                "ols": []
            }
        }

    def validate(
            self,
            validation_loader: DataLoader,
            validation_dir: str=""
        ):
        
        with torch.no_grad():
            for iter, data_dict in enumerate(validation_loader):
                print(f"{iter=}")
                # Get Data.
                data = data_dict['radar_data'].to(self.device)
                confmap_gt = data_dict['anno']['confmaps'].to(self.device)
                seq_name = data_dict['seq_names'][0]
                start_frame = data_dict['start_frame'].item()
                # Get Model Inference.
                confmap_pred = self.net(data.float())
                # Calculate Loss.
                loss_class_0 = self.criterion(confmap_pred[:,0,:,:,:], 
                                              confmap_gt[:,0,:,:,:].float()
                                            ).item()
                loss_class_1 = self.criterion(confmap_pred[:,1,:,:,:], 
                                              confmap_gt[:,1,:,:,:].float()
                                            ).item()
                loss_class_2 = self.criterion(confmap_pred[:,2,:,:,:], 
                                              confmap_gt[:,2,:,:,:].float()
                                            ).item()
                self.metrics["validation"]["class_0"]["loss"].append(loss_class_0)
                self.metrics["validation"]["class_1"]["loss"].append(loss_class_1)
                self.metrics["validation"]["class_2"]["loss"].append(loss_class_2)
                pred = confmap_pred[0,:,0,:,:]
                pred = pred.cpu().detach().numpy()

                result = post_process_single_frame(
                    pred, self.dataset, self.config_dict)
                
                seq_res_dir = os.path.join(validation_dir, seq_name)
                if not os.path.exists(seq_res_dir):
                    os.makedirs(seq_res_dir)
                f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'a')
                f.close()
                save_path = os.path.join(
                    validation_dir, seq_name, 'rod_res.txt')

                write_dets_results_single_frame(
                    result, start_frame, save_path, self.dataset)
                
                gt_path = os.path.join(
                    self.config_dict["dataset_cfg"]["anno_root"], 
                    f"train/{seq_name.upper()}.txt")
                data_path = os.path.join(
                    self.dataset.data_root, 
                    'sequences', 'train',
                    gt_path.split('/')[-1][:-4])
                n_frame = len(os.listdir(os.path.join(
                        data_path, 
                        self.dataset.sensor_cfg.camera_cfg['image_folder'])))
                gt_dets = read_gt_txt(gt_path, n_frame, self.dataset)
                sub_dets = read_rodnet_res(save_path, n_frame, self.dataset)
                # Continue if there are no detections.
                if sub_dets is None:
                    continue
                olss_all = {
                    (start_frame, catId): compute_ols_dts_gts(
                        gt_dets, sub_dets, start_frame, catId, self.dataset) \
                            for catId in range(3)}
                self.metrics["validation"]["ols"].append(olss_all)
        
    def save_metrics(self, results_dir: str=""):
        save_model_path = os.path.join(results_dir, "validation_metrics_base.json")
        with open(save_model_path, 'w') as fp:
            json.dump(self.metrics, fp)
