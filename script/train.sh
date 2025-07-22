#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi


dataset_name="$1"
scale="$2"
value_to_mask="$3"
dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# Gaussian Grouping training
python train.py    -s $dataset_folder -r ${scale}  -m output/${dataset_name} --config_file config/gaussian_dataset/train.json --value_to_mask ${value_to_mask}

# Segmentation rendering using trained model
python render.py -m output/${dataset_name} --num_classes 256
