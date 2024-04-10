# Radar Based Object Detection
## Cloning RODNet and CRUW directories
-  Change the current working directory to "radar_detections"
- clone the RODNet repository using: git clone https://github.com/yizhou-wang/RODNet.git
- clone the cruw-devkit repository using: git clone https://github.com/yizhou-wang/cruw-devkit.git
## Create a new conda environment
- Run the following lines of code in the terminal
    - conda create -n rodnet python=3.11 -y
    - conda activate rodnet
    - For MAC: pip3 install torch torchvision torchaudio
    - For Windows: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    - pip install -e .
    - cd cruw-devkit
    - pip install .
    - cd ..
    - pip install numpy

## Downloading the dataset
- Use the following google drive link to download the "ROD2024" folder as a subdirectory under "radar_detections": https://drive.google.com/drive/folders/1XXXKaU6_MAtqp9imyqpOEVu2vbvuqaCn?usp=sharing
- Move the downloaded ROD2024 folder inside the "radar_detections" main folder

## Configuration file
config_rodnet_cdc_win16.py is the file that contains the path to the dataset folder and other relevant variables, such as the number of epochs and batch size required by the custom CRUW dataset.

## Processing the dataset
- change the working directory to RODNet-master:
cd RODNet-master
- Run the follow line of code in the terminal: python tools/prepare_dataset/prepare_data.py --config ../config_rodnet_cdc_win16.py --data_root ../ROD2024 --split train --out_data_dir ../data_final_converted

## Running the final_project.ipynb
This is the main jupyter notebook used for the radar object detection.

- There are two ways to run this script:
    - Running the entire script which involves training and validation.
        - In the section Model Validation/testing. Comment the 3 and 4 statments and either uncomment 1 or 2 depending on the avaibility of GPU. For base model (rod_v0) and for the multibranch model (rod_v1).
    - Only running the validation and using the provided trained models as .pkl files
        - Download the "trained_models" folder containing .pkl files as a subdirectory under "radar_detections": https://drive.google.com/drive/folders/1GW93bPf7UZ-OhiEhrsuTuxwWYB4yqbbL?usp=sharing 
        - Skip the "Model training" section by not running the 8 cells inside it.
        - In the section Model Validation/testing. Comment the 1 and 2 statments and either uncomment 3 or 4 depending on the avaibility of GPU. For base model (rod_v0) and for the multibranch model (rod_v1).

## trainModel.py and getModel.py
The trainModel.py contains the function declaration for the training schedule. And getModel.py contains the class definition for the base model and the multibranch model.
