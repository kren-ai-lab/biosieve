#!/usr/bin/bash

sylphy get-embedding \
  --model facebook/esm2_t6_8M_UR50D \
  --input-data biosieve_example_dataset_1000.csv \
  --sequence-identifier sequence \
  --output demo_esm2_t6_8M_UR50D.csv \
  --device cuda --precision fp32 --batch-size 4
