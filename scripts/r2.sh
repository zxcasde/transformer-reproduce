#!/bin/bash

# cd "$(dirname "$0")/.."

python src/decoder2.py --model_dir ./models_new --max_epochs 50 --max_batches 30 --results_dir ./evaluation_results_6