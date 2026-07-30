#!/usr/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

python_bin="$repo_root/.venv/bin/python"
export PATH="$repo_root/.venv/bin:$PATH"

cp -R "$repo_root/examples" "$tmpdir/"

cd "$tmpdir/examples/raw_data_examples"

"$python_bin" ../generating_random_datasets.py
"$python_bin" ../generate_demo_descriptors.py
"$python_bin" ../generate_demo_distances_structures.py
"$python_bin" ../prepare_cluster_mapping.py
"$python_bin" ../prepare_embedding_for_reductions.py

bash ../testing_splitters.sh
bash ../testing_reducers.sh
