#!/usr/bin/bash
set -euo pipefail

# Run from examples/raw_data_examples/

# Exact deduplication
biosieve reduce \
  -i biosieve_example_dataset_1000.csv \
  -o data_nr_exact.csv \
  --strategy exact \
  --mapping-output map_exact.csv \
  --report-output report_exact.json

# Identity greedy
biosieve reduce \
  -i biosieve_example_dataset_1000.csv \
  -o data_nr_identity.csv \
  --strategy identity_greedy \
  --mapping-output map_identity.csv \
  --report-output report_identity.json

# K-mer Jaccard
biosieve reduce \
  -i biosieve_example_dataset_1000.csv \
  -o data_nr_kmer.csv \
  --strategy kmer_jaccard \
  --mapping-output map_kmer.csv \
  --report-output report_kmer.json

# MMseqs2
biosieve reduce \
  -i biosieve_example_dataset_1000.csv \
  -o data_nr_mmseqs2.csv \
  --strategy mmseqs2 \
  --mapping-output map_mmseqs2.csv \
  --report-output report_mmseqs2.json

# Embedding cosine distance
biosieve reduce \
  -i biosieve_example_dataset_1000.csv \
  -o data_nr_emb_cosine.csv \
  --strategy embedding_cosine \
  --mapping-output map_emb_cosine.csv \
  --report-output report_emb_cosine.json \
  --params ../configs/params_reducer.yaml

# Descriptor Euclidean distance
biosieve reduce \
  -i biosieve_example_dataset_1000_desc.csv \
  -o data_nr_desc_euclidean.csv \
  --strategy descriptor_euclidean \
  --mapping-output map_desc_euclidean.csv \
  --report-output report_desc_euclidean.json

# MinHash Jaccard (approximate, requires datasketch: pip install biosieve[minhash])
biosieve reduce \
  -i biosieve_example_dataset_1000.csv \
  -o data_nr_minhash.csv \
  --strategy minhash_jaccard \
  --mapping-output map_minhash.csv \
  --report-output report_minhash.json \
  --params ../configs/params_reducer_minhash_jaccard.yaml

# Structural distance
biosieve reduce \
  -i biosieve_example_dataset_1000.csv \
  -o data_nr_structural.csv \
  --strategy structural_distance \
  --mapping-output map_structural.csv \
  --report-output report_structural.json
