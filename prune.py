
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
import torch.nn.utils.prune as prune

def print_sparsity(
        rodnet: Union[
            RODNetCDC, 
            RODNetHG, 
            RODNetHGwI, 
            RODNetCDCDCN, 
            RODNetHGDCN, 
            RODNetHGwIDCN],
    ):
    """
    Print the final sparsities of the pruned model.

    Parameters
    ----------
        rodnet: Type[nn.Module]
            This is a specific RodNet architecture depending on the model
            configurations.
    """
    print(
        "Sparsity in conv1a.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.encoder.conv1a.weight == 0))
            / float(rodnet.cdc.encoder.conv1a.weight.nelement())
        )
    )
    print(
        "Sparsity in conv1b.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.encoder.conv1b.weight == 0))
            / float(rodnet.cdc.encoder.conv1b.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2a.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.encoder.conv2a.weight == 0))
            / float(rodnet.cdc.encoder.conv2a.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2b.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.encoder.conv2b.weight == 0))
            / float(rodnet.cdc.encoder.conv2b.weight.nelement())
        )
    )
    print(
        "Sparsity in conv3a.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.encoder.conv3a.weight == 0))
            / float(rodnet.cdc.encoder.conv3a.weight.nelement())
        )
    )
    print(
        "Sparsity in conv3b.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.encoder.conv3b.weight == 0))
            / float(rodnet.cdc.encoder.conv3b.weight.nelement())
        )
    )
    print(
        "Sparsity in convt1.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.decoder.convt1.weight == 0))
            / float(rodnet.cdc.decoder.convt1.weight.nelement())
        )
    )
    print(
        "Sparsity in convt2.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.decoder.convt2.weight == 0))
            / float(rodnet.cdc.decoder.convt2.weight.nelement())
        )
    )
    print(
        "Sparsity in convt3.weight: {:.2f}%".format(
            100. * float(torch.sum(rodnet.cdc.decoder.convt3.weight == 0))
            / float(rodnet.cdc.decoder.convt3.weight.nelement())
        )
    )
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(rodnet.cdc.encoder.conv1a.weight == 0)
                + torch.sum(rodnet.cdc.encoder.conv1b.weight == 0)
                + torch.sum(rodnet.cdc.encoder.conv2a.weight == 0)
                + torch.sum(rodnet.cdc.encoder.conv2b.weight == 0)
                + torch.sum(rodnet.cdc.encoder.conv3a.weight == 0)
                + torch.sum(rodnet.cdc.encoder.conv3b.weight == 0)
                + torch.sum(rodnet.cdc.decoder.convt1.weight == 0)
                + torch.sum(rodnet.cdc.decoder.convt2.weight == 0)
                + torch.sum(rodnet.cdc.decoder.convt3.weight == 0)
            )
            / float(
                rodnet.cdc.encoder.conv1a.weight.nelement()
                + rodnet.cdc.encoder.conv1b.weight.nelement()
                + rodnet.cdc.encoder.conv2a.weight.nelement()
                + rodnet.cdc.encoder.conv2b.weight.nelement()
                + rodnet.cdc.encoder.conv3a.weight.nelement()
                + rodnet.cdc.encoder.conv3b.weight.nelement()
                + rodnet.cdc.decoder.convt1.weight.nelement()
                + rodnet.cdc.decoder.convt2.weight.nelement()
                + rodnet.cdc.decoder.convt3.weight.nelement()
            )
        )
    )

def multi_parameter_unstructured_pruning(
        rodnet: Union[
            RODNetCDC, 
            RODNetHG, 
            RODNetHGwI, 
            RODNetCDCDCN, 
            RODNetHGDCN, 
            RODNetHGwIDCN],
        amount_3d_conv: float,
        amount_3d_trans_conv: float
    ):
    """
    Local pruning or pruning tensors in a model one by one per layer. Removal
    of percent amount of connections of each 3D convolutional layer or FC layers.

    Parameters
    ----------
        rodnet: Type[nn.Module]
            This is a specific RodNet architecture depending on the model
            configurations.

        amount_3d_conv: float
            The sparsity amount to prune the 3D convolutional layers.

        amount_3d__trans_conv: float
            The sparsity amount to prune the 3D transpose convolutional layers.
    """
    for name, module in rodnet.named_modules():
        # Prune amount_3d_conv % of connections in all 3D-conv layers
        if isinstance(module, torch.nn.Conv3d):
            prune.l1_unstructured(module, name='weight', amount=amount_3d_conv)
            prune.remove(module, "weight")
        # Prune amount_3d_trans_conv % of connections in all ConvTranspose3d layers.
        elif isinstance(module, torch.nn.ConvTranspose3d):
            prune.l1_unstructured(module, name='weight', amount=amount_3d_trans_conv)
            prune.remove(module, "weight")
    return rodnet

def global_unstructured_pruning(
        rodnet: Union[
            RODNetCDC, 
            RODNetHG, 
            RODNetHGwI, 
            RODNetCDCDCN, 
            RODNetHGDCN, 
            RODNetHGwIDCN],
        amount: float=0.2
    ):
    """
    Prune the model all at once, the lowest % amount of connections 
    across the whole model.

    Parameters
    ----------
        rodnet: Type[nn.Module]
            This is a specific RodNet architecture depending on the model
            configurations.

        amount: float
            The sparsity amount to prune the model.
    """
    parameters = (
        (rodnet.cdc.encoder.conv1a, "weight"),
        (rodnet.cdc.encoder.conv1b, "weight"),
        (rodnet.cdc.encoder.conv2a, "weight"),
        (rodnet.cdc.encoder.conv2b, "weight"),
        (rodnet.cdc.encoder.conv3a, "weight"),
        (rodnet.cdc.encoder.conv3b, "weight"),
        (rodnet.cdc.decoder.convt1, "weight"),
        (rodnet.cdc.decoder.convt2, "weight"),
        (rodnet.cdc.decoder.convt3, "weight")
    )
    prune.global_unstructured(
        parameters=parameters,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    prune.remove(rodnet.cdc.encoder.conv1a, "weight")
    prune.remove(rodnet.cdc.encoder.conv2a, "weight")
    prune.remove(rodnet.cdc.encoder.conv2b, "weight")
    prune.remove(rodnet.cdc.encoder.conv3a, "weight")
    prune.remove(rodnet.cdc.encoder.conv3b, "weight")
    prune.remove(rodnet.cdc.decoder.convt1, "weight")
    prune.remove(rodnet.cdc.decoder.convt2, "weight")
    prune.remove(rodnet.cdc.decoder.convt3, "weight")
    return rodnet