#!/usr/bin/bash
set -euo pipefail

# Run from examples/raw_data_examples/

# Random
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_random \
  --strategy random \
  --config ../configs/params_split_random.yaml

# Stratified
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_stratified \
  --strategy stratified \
  --config ../configs/params_split_stratified.yaml

# Group
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_group \
  --strategy group \
  --config ../configs/params_split_group.yaml

# Time-based
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_time \
  --strategy time \
  --config ../configs/params_split_time.yaml

# Distance aware (embeddings)
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_distance_emb \
  --strategy distance_aware \
  --embeddings embeddings.npy \
  --embedding-ids embedding_ids.csv \
  --config ../configs/params_split_distance_aware.yaml

# Distance aware (descriptors)
biosieve split \
  -i biosieve_example_dataset_1000_desc.csv \
  -o runs/split_distance_desc \
  --strategy distance_aware \
  --config ../configs/params_split_distance_aware_descriptors.yaml

# Homology aware (without previous results)
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_homology \
  --strategy homology_aware \
  --config ../configs/params_split_homology_aware.yaml

# Homology aware (precomputed data)
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_homology_precomputed \
  --strategy homology_aware \
  --clusters runs/mmseqs2_clusters/clusters_cluster.tsv \
  --config ../configs/params_split_homology_aware_precomputed.yaml

# Cluster-aware
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_cluster_aware \
  --strategy cluster_aware \
  --config ../configs/params_split_cluster_aware.yaml

# Cluster-aware with external mapping file
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_cluster_aware_mapped \
  --strategy cluster_aware \
  --config ../configs/params_split_cluster_aware_with_mapping.yaml

# Stratified numerical
biosieve split \
  -i biosieve_example_dataset_1000_desc.csv \
  -o runs/split_stratified_num \
  --strategy stratified_numeric \
  --config ../configs/params_split_stratified_numerical.yaml

# Random k-fold
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_random_kfold \
  --strategy random_kfold \
  --config ../configs/params_split_random_kfold.yaml

# Stratified k-fold
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_stratified_kfold \
  --strategy stratified_kfold \
  --config ../configs/params_split_stratified_kfold.yaml

# Group k-fold
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_group_kfold \
  --strategy group_kfold \
  --config ../configs/params_split_group_kfold.yaml

# Stratified numerical k-fold
biosieve split \
  -i biosieve_example_dataset_1000_desc.csv \
  -o runs/split_stratified_num_kfold \
  --strategy stratified_numeric_kfold \
  --config ../configs/params_split_stratified_num_kfold.yaml

# Distance aware k-fold (embeddings)
biosieve split \
  -i biosieve_example_dataset_1000.csv \
  -o runs/split_distance_emb_kfold \
  --strategy distance_aware_kfold \
  --embeddings embeddings.npy \
  --embedding-ids embedding_ids.csv \
  --config ../configs/params_split_distance_aware_embedding_kfold.yaml

# Distance aware k-fold (descriptors)
biosieve split \
  -i biosieve_example_dataset_1000_desc.csv \
  -o runs/split_distance_desc_kfold \
  --strategy distance_aware_kfold \
  --config ../configs/params_split_distance_aware_desc_kfold.yaml
