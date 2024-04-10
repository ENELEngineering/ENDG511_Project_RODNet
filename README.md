# ENDG511 Project

This repository shows the implementation of the concepts Tiny and Fast AI using
PyTorch, Tensorflow, and YoloV5 to continue the work behind 
*Radar Object Detection [RODNet](https://github.com/yizhou-wang/RODNet).* and aswell
as provide performance comparison between radar-based object detection and 
conventional camera-based object detection in cases of low visibility.

# Overview

- [Problem/Use Case](#problemuse-case)
- [Setup](#setup)
    - [Dataset Partitioning](#dataset-partitioning)
    - [Configurations](#configurations)
- [Results and Analysis](#results-and-analysis)
- [Sample Commands](#sample-commands)
- [References](#references)

# Problem/Use Case

There is a limitation of utilizing conventional camera-based sensing object detection networks under conditions of weak/strong lighting during night operations or very bright environments or perhaps poor weather conditions such as fog, smoke, dust, storms, and rain which can lead to little/high exposure or blur/occluded images.

The intention of utilizing radar-based detection on top of camera-based detections is to take advantage of radar’s increase in reliability to sense objects during harsh weather conditions. Frequency modulated continuous wave (FMCW) radar operates in the millimeter-wave (MMW) band (30-300GHz) which is lower than visible light, but gives it properties of increase penetration through fog, smoke, and dust and increased range detection capabilities due to the large bandwidth and increased working frequency.

The increasing trend of autonomous vehicle operations provides the prospects of utilizing radar-based object detection technology to aid in smarter decisions in autonomous vehicles by providing more accurate detections under poor visibility conditions where camera-based sensing accuracy could decline. 

The motivations behind this project is to modify the RODNet architectures to implement the concepts learned in the course for reducing the model size but maintaining its accuracy such as pruning, quantization, and clustering. Secondly to increase the model's speed by implementing a multibranching architecture based around RODNet.  

# Background

## Challenges

As described in the project report, there were challenges that was encountered during the 
pruning and quantization of the base RODNet architecture using PyTorch. The source
code showing these efforts are under /pytorch_challenges. 

The subdirectory /primary_effort shows the initial efforts or attempts at pruning 
and quantizing the RODNet model.

The subdirectory /secondary_effort shows the second attempt at pruning and 
quantizing the RODNet model.

Our future work will also
involve revisiting this effort in hopes to modify to perform proper pruning and 
quantization of the model that yeilds expected results. 

## 

# Setup

This section shows the steps for setting up the project to make use of the scripts provided.

1) Clone the [RODNet Repository](https://github.com/yizhou-wang/RODNet). 
2) [Install Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).
3) Follow the RODNet repository's installation guidelines. However, encountered a few issues following the guidelines for installing PyTorch with Cuda. This [guideline](https://medium.com/@harunijaz/a-step-by-step-guide-to-installing-cuda-with-pytorch-in-conda-on-windows-verifying-via-console-9ba4cd5ccbef) for installing PyTorch with Cuda worked seamlessly.
 
The following environment was used for testing. 

**Environment**

* Python: 3.11.8
* Cuda: 11.8
* torch: 2.2.1

Verify that Cuda is installed and PyTorch recognizes this installation.

```python
>> print(torch.cuda.is_available())
>> True
```

Once the repository is cloned, the scripts provided were copied under `/tools/` of the RODNet repository.

## Dataset Partitioning

We ran into difficulties testing the model which was suspected to be based around missing test annotations in the provided RODNet dataset. The [issue](https://github.com/yizhou-wang/RODNet/issues/78) was also raised in the RODNet repository.

The actual error that was generated when running test.py for model testing was:

```shell
Length of testing data: 111
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "\RODNet\tools\test.py", line 184, in <module>
    data = data_dict['radar_data']
           ~~~~~~~~~^^^^^^^^^^^^^^
KeyError: 'radar_data'
```

However, replacing the testing samples with training samples resolved the issue. The decision was then to partition the training samples into train and test since the training samples provided were complete that allowed to run the three tasks: train, test, eval.

The final partitioned dataset structure is as follows:

```shell
dataset base directory
        |---calib
        |     |-----2019_05_09
        |     |-----2019_09_29
        |---annotations
        |     |-----test
        |     |-----train
        |---sequences
        |     |-----test
        |             |----2019_04_30_MLMS002
        |             |----2019_09_29_ONRD006
        |     |-----train
        |             |----2019_04_30_PM2S004
        |             |----2019_04_09_BMS1000
        |             |----2019_04_30_PBMS002
        |             |----2019_04_30_PCMS001
        |             |----2019_04_30_MLMS000
        |             |----2019_05_29_PBMS007
        |             |----2019_04_09_PMS1000
        |             |----2019_05_29_BCMS000
        |             |----2019_09_29_ONRD002
        |             |----2019_05_09_CM1S004
```

## Configurations

The configuration final used throughout this project is: `/configs/config_rodnet_cdc_win16.py`.

This file was edited based on the changes in the dataset.

```python
dataset_cfg = dict(
    dataset_name='ROD2021',
    base_root="path to the base directory of the dataset",
    data_root="path to the sequences directory",
    anno_root="path to the annotations directory",
    anno_ext='.txt',
    train=dict(
        subdir='train',
        # seqs=[],  # can choose from the subdir folder
    ),
    valid=dict(
        subdir='valid',
        seqs=[],
    ),
    test=dict(
        subdir='test',
        # seqs=[],  # can choose from the subdir folder
    ),
    demo=dict(
        subdir='demo',
        seqs=[],
    ),
)
```

```python
train_cfg = dict(
    n_epoch=10,
    batch_size=4,
    lr=0.00001,
    lr_step=5,  # lr will decrease 10 times after lr_step epoches
    win_size=16,
    train_step=1,
    train_stride=4,
    log_step=100,
    save_step=10000,
)
```

# Results and Analysis

[TODO]: Provide link to point to a location storing the saved model, results, and dataset used.


# Sample Commands


# References

* The work done in this repository [RODNet](https://github.com/yizhou-wang/RODNet/tree/master) was continued in this project to implement multibranching, pruning, quantization, and clustering methods.