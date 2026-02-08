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

# Homology aware (without previous results)
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_homology \
  --strategy homology_aware \
  --params ../../configs/params_split_homology_aware.yaml

# Homology aware (precomputed data)
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_homology \
  --strategy homology_aware \
  --params ../../configs/params_split_homology_aware_precomputed.yaml

# Cluster-aware simple
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_clusteraware \
  --strategy cluster_aware \
  --params ../../configs/params_split_cluster_aware.yaml

# Cluster-aware with file for mapping clusters
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_clusteraware \
  --strategy cluster_aware \
  --params ../../configs/params_split_cluster_aware_with_mapping.yaml

# Stratified-numerical
biosieve split \
  --in biosieve_example_dataset_1000_desc.csv \
  --outdir runs/split_stratnum \
  --strategy stratified_numeric \
  --params ../../configs/params_split_stratified_numerical.yaml

# Random K-Fold
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_random_kfold \
  --strategy random_kfold \
  --params ../../configs/params_split_random_kfold.yaml

# Stratified K-fold
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_stratified_kfold \
  --strategy stratified_kfold \
  --params ../../configs/params_split_stratified_kfold.yaml

# Group K-Fold
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_group_kfold \
  --strategy group_kfold \
  --params ../../configs/params_split_group_kfold.yaml

# Stratified numerical k-fold
biosieve split \
  --in biosieve_example_dataset_1000_desc.csv \
  --outdir runs/split_stratnum_kfold \
  --strategy stratified_numeric_kfold \
  --params ../../configs/params_split_stratified_num_kfold.yaml

# Distance aware embedding k-fold
biosieve split \
  --in biosieve_example_dataset_1000.csv \
  --outdir runs/split_distanceaware_kfold \
  --strategy distance_aware_kfold \
  --params ../../configs/params_split_distance_aware_embedding_kfold.yaml
