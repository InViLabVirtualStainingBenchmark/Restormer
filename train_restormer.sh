#!/bin/bash
CONFIG=${1:-/code/Restormer/experiments/BCI_Restormer/train_BCI.yml}
export PYTHONPATH=/code/Restormer/basicsr:/code/Restormer:/usr/local/lib64/python3.9/site-packages:$PYTHONPATH
cd /code/Restormer
python3 basicsr/train.py \
    -opt $CONFIG \
    --auto_resume \
    2>&1 | tee /output/train_log.txt
