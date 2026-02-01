#!/usr/bin/bash

biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_random \
  --strategy random \
  --params ../../configs/params_split_random.yaml

biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_stratified \
  --strategy stratified \
  --params ../../configs/params_split_stratified.yaml