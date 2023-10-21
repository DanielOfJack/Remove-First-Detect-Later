#!/bin/bash

# Check if sufficient arguments are passed
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <seed> <dataset> <labels> <epochs>"
    exit 1
fi

# Get arguments
seed=$1
dataset=$2
labels=$3
epochs=$4

# Loop over the models
for model in "UNET" "RNET6" "RFDL"; do
    # Check the model to set the appropriate loss
    if [ "$model" == "RNET6" ]; then
        loss="mse"
    else
        loss="binary_crossentropy"
    fi
    
    #Run CDAE Training if RFDL
    if [ "$model" == "RFDL" ]; then
        python experiment/train_CDAE.py --dataset="$dataset"
    fi
    
    if [ "$labels" == "Transfer" ]; then
        python experiment/run_trial.py $model --save_name="$model"_"$seed" --loss=$loss --dataset="$dataset" --labels="AOFlagger" --seed=$seed --epochs=50 --report=0
        
         python experiment/run_transfer.py $model --save_name="$model"_"$seed" --loss=$loss --dataset="$dataset" --labels="$labels" --seed=$seed --epochs="$epochs"
         
    else
        python experiment/run_trial.py $model --save_name="$model"_"$seed" --loss=$loss --dataset="$dataset" --labels="$labels" --seed=$seed --epochs="$epochs"
        
    fi
    
done

