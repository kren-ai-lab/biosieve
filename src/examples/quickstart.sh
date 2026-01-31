#!/usr/bin/bash

#Exact dedup
biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out data_nr_exact.csv \
  --strategy exact \
  --map redundancy_map_exact.csv \
  --report reduction_exact.json

#Random split
biosieve split \
  --in data_nr_exact.csv \
  --outdir splits_random \
  --strategy random \
  --seed 13

#Stratified split
biosieve split \
  --in data_nr_exact.csv \
  --outdir splits_stratified \
  --strategy stratified \
  --seed 13

biosieve split \
  --in data_nr_exact.csv \
  --outdir splits_random_2 \
  --strategy random \
  --seed 13