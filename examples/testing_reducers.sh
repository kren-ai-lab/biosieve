#!/usr/bin/bash

#Exact dedup
biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out data_nr_exact.csv \
  --strategy exact \
  --map redundancy_map_exact.csv \
  --report reduction_exact.json

# identity_greedy
biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out data_nr_identity.csv \
  --strategy identity_greedy \
  --map map_identity.csv \
  --report reduction_identity.json

# kmer_jaccard
biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out data_nr_kmer.csv \
  --strategy kmer_jaccard \
  --map map_kmer.csv \
  --report reduction_kmer.json

# mmseq2
biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out data_nr_mmseqs2.csv \
  --strategy mmseqs2 \
  --map map_mmseqs2.csv \
  --report reduction_mmseqs2.json

# embedding cosine-distance
biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out data_nr_embcos.csv \
  --strategy embedding_cosine \
  --map map_embcos.csv \
  --report reduction_embcos.json

# descriptors euclidean distance
biosieve reduce \
  --in biosieve_example_dataset_1000_desc.csv \
  --out data_nr_deuc.csv \
  --strategy descriptor_euclidean \
  --map map_deuc.csv \
  --report reduction_deuc.json

# structural distances
biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out data_nr_struct.csv \
  --strategy structural_distance \
  --map map_struct.csv \
  --report reduction_struct.json
