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

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(model_name: str, epochs: list, losses: list, labels: list):
    """
    Parameters
    ----------
        loss: list
            Ensure this contains [[loss_array], [loss_array], ...]
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for loss, label in zip(losses, labels):
        ax.plot(epochs, loss, linewidth=1, label=f"{label} Loss")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f'{model_name} Training Loss vs. Epoch')
    return fig

def plot_base_pdf(model_name: str, metrics: dict):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.hist(np.round(
        np.array(metrics["validation"]["class_0"]["loss"]),3), 
        bins=30, color='red', alpha=0.5, label='pedestrian',density=True)
    ax.hist(np.round(
        np.array(metrics["validation"]["class_1"]["loss"]),3), 
        bins=30, color='green', alpha=0.5, label='bicycle',density=True)
    ax.hist(np.round(
        np.array(metrics["validation"]["class_2"]["loss"]),3), 
        bins=30, color='blue', alpha=0.5, label='car',density=True)
    
    ax.set_title(f'{model_name} Loss PDF')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper left', ncol=3)
    ax.grid(True)
    return fig

def plot_branch_pdf(model_name: str, metrics: dict):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 6), tight_layout=True)
    ax[0].hist(np.round(
        np.array(metrics["validation"]["short"]["class_0"]["loss"]),3), 
        bins=30, color='red', alpha=0.5, label='pedestrian',density=True)
    ax[0].hist(np.round(
        np.array(metrics["validation"]["short"]["class_1"]["loss"]),3), 
        bins=30, color='green', alpha=0.5, label='bicycle',density=True)
    ax[0].hist(np.round(
        np.array(metrics["validation"]["short"]["class_2"]["loss"]),3), 
        bins=30, color='blue', alpha=0.5, label='car',density=True)
    
    ax[1].hist(np.round(
        np.array(metrics["validation"]["long"]["class_0"]["loss"]),3), 
        bins=30, color='red', alpha=0.5, label='pedestrian',density=True)
    ax[1].hist(np.round(
        np.array(metrics["validation"]["long"]["class_1"]["loss"]),3), 
        bins=30, color='green', alpha=0.5, label='bicycle',density=True)
    ax[1].hist(np.round(
        np.array(metrics["validation"]["long"]["class_2"]["loss"]),3), 
        bins=30, color='blue', alpha=0.5, label='car',density=True)
    
    ax[0].set_title(f'{model_name} Loss PDF Short Branch')
    ax[1].set_title(f'{model_name} Loss PDF Long Branch')
    ax[0].set_xlabel('Loss')
    ax[1].set_xlabel('Loss')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_ylabel('Probability Density')
    ax[0].legend(loc='upper left', ncol=3)
    ax[1].legend(loc='upper left', ncol=3)
    ax[0].grid(True)
    ax[1].grid(True)
    return fig
 