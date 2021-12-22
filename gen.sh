#!/bin/bash
MODE="valid" 
NAME="only_copy"
module load anaconda
source activate 409B
python gen.py --input_path data/cut_${MODE}.txt --output_path data/gen_${NAME}_${MODE}.txt
python gen_json_for_eval.py --format_path data/${MODE}.jsonl --input_path data/gen_${NAME}_${MODE}.txt --output_path data/gen_${NAME}_${MODE}.jsonl
python eval.py data/gen_${NAME}_${MODE}.jsonl data/${MODE}.jsonl
