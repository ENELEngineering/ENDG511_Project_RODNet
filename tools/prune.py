
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
    
from rodnet.utils.load_configs import (
    load_configs_from_file, 
    parse_cfgs, 
    update_config_dict
)
from cruw import CRUW
import argparse
import torch
import torch.nn.utils.prune as prune
import os

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
        # Prune amount_3d_trans_conv % of connections in all ConvTranspose3d layers.
        elif isinstance(module, torch.nn.ConvTranspose3d):
            prune.l1_unstructured(module, name='weight', amount=amount_3d_trans_conv)

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
    print_sparsity(rodnet)

def check_path(file_path: str) -> str:
    """
    Checks if the path exists or not.

    Parameters
    ----------
        file_path: str
            This is the path to check.

    Returns
    -------
        file_path: str
            If this path exists, then it is returned.
    """
    if file_path is not None and os.path.exists(file_path):
        return file_path
    else:
        raise ValueError(f"The following path does not exist: {file_path}")

def load_model(checkpoint_path: str) -> dict:
    """
    Loads a PKL model file path.

    Parameters
    ----------
        checkpoint_path: str
            This is the PKL path to load torch.

    Returns
    -------
        checkpoint: dict
            This is a loaded pkl file stored as a dictionary.
    """
    return torch.load(checkpoint_path)

def build_model(
        config_dict: dict, 
        dataset: CRUW, 
        use_noise_channel: bool, 
        checkpoint: dict
    ):
    """
    Builds the model architecture based on configurations.

    Parameters
    ----------
        config_dict: dict
            The model configurations
        
        dataset: CRUW
            Dataset configurations used to train the model.

        use_noise_channel: bool
            Specification to use noise channel from the command line.

        checkpoint: dict
            The loaded pkl model file from training.

    Returns
    -------
        rodnet: Type[nn.Module]
            This is a specific RodNet architecture depending on the model
            configurations.
    """
    radar_configs = dataset.sensor_cfg.radar_cfg
    n_class = dataset.object_cfg.n_class

    if use_noise_channel:
        n_class_test = n_class + 1
    else:
        n_class_test = n_class

    model_cfg = config_dict['model_cfg']
    if 'stacked_num' in model_cfg:
        stacked_num = model_cfg['stacked_num']
    else:
        stacked_num = None

    if model_cfg['type'] == 'CDC':
        from rodnet.models import RODNetCDC
        rodnet = RODNetCDC(in_channels=2, n_class=n_class_test).cuda()
    elif model_cfg['type'] == 'HG':
        from rodnet.models import RODNetHG
        rodnet = RODNetHG(
            in_channels=2, 
            n_class=n_class_test, 
            stacked_num=stacked_num).cuda()
    elif model_cfg['type'] == 'HGwI':
        from rodnet.models import RODNetHGwI
        rodnet = RODNetHGwI(
            in_channels=2, 
            n_class=n_class_test, 
            stacked_num=stacked_num).cuda()
    elif model_cfg['type'] == 'CDCv2':
        from rodnet.models import RODNetCDCDCN
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNetCDCDCN(
            in_channels=in_chirps, 
            n_class=n_class_test,            
            mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
            dcn=config_dict['model_cfg']['dcn']).cuda()
    elif model_cfg['type'] == 'HGv2':
        from rodnet.models import RODNetHGDCN
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNetHGDCN(
            in_channels=in_chirps, 
            n_class=n_class_test, 
            stacked_num=stacked_num,
            mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
            dcn=config_dict['model_cfg']['dcn']).cuda()
    elif model_cfg['type'] == 'HGwIv2':
        from rodnet.models import RODNetHGwIDCN
        in_chirps = len(radar_configs['chirp_ids'])
        rodnet = RODNetHGwIDCN(
            in_channels=in_chirps, 
            n_class=n_class_test, 
            stacked_num=stacked_num,
            mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
            dcn=config_dict['model_cfg']['dcn']).cuda()
    else:
        raise NotImplementedError(
            f"The following model type is not supported: {model_cfg['type']}")
    
    if 'optimizer_state_dict' in checkpoint:
        rodnet.load_state_dict(checkpoint['model_state_dict'])
    else:
        rodnet.load_state_dict(checkpoint)
    rodnet.eval()
    return rodnet

def main():
    """
    Main program starting point.
    """
    parser = argparse.ArgumentParser(
        description='Prune RODNet.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        help='choose rodnet model configurations'
    )
    parser.add_argument(
        '--sensor_config', 
        type=str, 
        default='sensor_config_rod2021'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        help='path to the saved trained model'
    )
    parser.add_argument(
        '--res_dir', 
        type=str, 
        default='./results/', 
        help='directory to save testing results'
    )
    parser.add_argument(
        '--use_noise_channel', 
        action="store_true", 
        help="use noise channel or not"
    )
    parser = parse_cfgs(parser)
    args = parser.parse_args()

    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # Update configs by args.

    dataset = CRUW(
        data_root=config_dict['dataset_cfg']['base_root'], 
        sensor_config_name=args.sensor_config)

    """
    # The following code is for dataset testing purposes.
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    dataset_configs = config_dict['dataset_cfg']
    train_configs = config_dict['train_cfg']
    test_configs = config_dict['test_cfg']
    win_size = train_configs['win_size']
    """
    
    checkpoint_path = check_path(args.checkpoint)
    checkpoint = load_model(checkpoint_path)
    rodnet = build_model(config_dict, dataset, args.use_noise_channel, checkpoint)
    
    global_unstructured_pruning(rodnet, amount=0.2)
    #print(list(rodnet.cdc.encoder.conv1a.named_parameters()))

if __name__ == '__main__':
    main()