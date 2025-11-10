#!/bin/bash

cd "$(dirname "$0")/.."

for i in {1..50}
do   
    python src/generate.py --epoch $i >> scripts/generate.txt 2>&1
done
