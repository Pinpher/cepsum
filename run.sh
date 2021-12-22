#!/bin/bash
module load anaconda
source activate 409B
python train.py --name only_copy_final --resume model/model_only_copy_8
