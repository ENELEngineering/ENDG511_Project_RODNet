# Copyright 2024. All Rights Reserved.
#
# This source code is provided solely for runtime interpretation by Python.
# 
# This python file is used explicitly to meet the project requirements provided
# in ENDG 511 at the University of Calgary.

from __future__ import annotations
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from rodnet.models import (
        RODNetCDC, 
        RODNetHG, 
        RODNetHGwI, 
        RODNetCDCDCN, 
        RODNetHGDCN, 
        RODNetHGwIDCN
    )
    
import torch

def dynamic_quantization(
        rodnet: Union[
            RODNetCDC, 
            RODNetHG, 
            RODNetHGwI, 
            RODNetCDCDCN, 
            RODNetHGDCN, 
            RODNetHGwIDCN],
):
    """
    Performs dynamic quantization of the model.

    Parameters
    ----------
        rodnet: Type[nn.Module]
            This is a specific RodNet architecture depending on the model
            configurations.

    Returns
    -------
        rodnet: Type[nn.Module]
            The model with the layers dynamically quantized.
    """
    quantized_model = torch.quantization.quantize_dynamic(
        rodnet,
        dtype=torch.quint8
    )
    # The code below was unsuccessful as the intention was to separately
    # quantize the two halves of the model architecture.

    # quantized_encoder = torch.quantization.quantize_dynamic(
    #     rodnet.cdc.encoder,
    #     {torch.nn.Conv3d},
    #     dtype=torch.qint8
    # )
    # rodnet.cdc.encoder = quantized_encoder

    # quantized_decoder = torch.quantization.quantize_dynamic(
    #     rodnet.cdc.decoder,
    #     {torch.nn.ConvTranspose3d},
    #     dtype=torch.qint8
    # )
    # rodnet.cdc.decoder = quantized_decoder

    return quantized_model

def static_quantization(
        rodnet: Union[
            RODNetCDC, 
            RODNetHG, 
            RODNetHGwI, 
            RODNetCDCDCN, 
            RODNetHGDCN, 
            RODNetHGwIDCN],
        config_dict: dict
):
    """
    Performs static quantization of the model.

    Parameters
    ----------
        rodnet: Type[nn.Module]
            This is a specific RodNet architecture depending on the model
            configurations.

    Returns
    -------
        rodnet: Type[nn.Module]
            The model with the layers statically quantized.
    """
    # Fuse layers and prepare the model for quantization
    qmodel = torch.quantization.fuse_modules(
        rodnet, 
        [
        ["cdc.encoder.conv1a", 'cdc.encoder.bn1a'], 
        ["cdc.encoder.conv1b", 'cdc.encoder.bn1b'],
        ["cdc.encoder.conv2a", 'cdc.encoder.bn2a'],
        ["cdc.encoder.conv2b", 'cdc.encoder.bn2b'],
        ["cdc.encoder.conv3a", 'cdc.encoder.bn3a'],
        ["cdc.encoder.conv3b", 'cdc.encoder.bn3b']
        ]
    )

    qmodel.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # Choose quantization configuration
    qmodel = torch.quantization.prepare(qmodel.cdc.encoder, inplace=True)

    # Calibrate the model (optional)
    # You can use your own calibration dataset for more accurate quantization
    # calibrate() function is used for dynamic quantization
    # For static quantization, you can use torch.quantization.convert(model)
    #calibrated_model = torch.quantization.calibrate(qmodel, config_dict)
    
    # Convert the calibrated model to a quantized model
    quantized_model = torch.quantization.convert(qmodel)
    rodnet.cdc.encoder = quantized_model
    return rodnet
    