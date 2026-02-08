biosieve reduce \
  --in biosieve_example_dataset_1000.csv \
  --out results/embcos/data_nr.csv \
  --map results/embcos/map.csv \
  --report results/embcos/report.json \
  --strategy embedding_cosine \
  --params ../../configs/demo_params_reducer.yaml
