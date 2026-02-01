#!/usr/bin/bash

# Random
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_random \
  --strategy random \
  --params ../../configs/params_split_random.yaml

# Stratified
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_stratified \
  --strategy stratified \
  --params ../../configs/params_split_stratified.yaml

# Group
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_group \
  --strategy group \
  --params ../../configs/params_split_group.yaml

# Time-based
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_time \
  --strategy time \
  --params ../../configs/params_split_time.yaml

# Distance aware (embedding)
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_distance_emb \
  --strategy distance_aware \
  --params ../../configs/params_split_distance_aware.yaml


# Distance aware (descriptors)
biosieve split \
  --in biosieve_example_dataset_1000_desc.csv \
  --outdir runs/split_distance_desc \
  --strategy distance_aware \
  --params ../../configs/params_split_distance_aware_descriptors.yaml
