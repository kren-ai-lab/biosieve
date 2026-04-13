import random
from datetime import date, timedelta
import math
import pandas as pd

def random_peptide(rng: random.Random, length: int) -> str:
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(rng.choice(aa) for _ in range(length))

def mutate_sequence(rng: random.Random, seq: str, n_mut: int = 1) -> str:
    aa = "ACDEFGHIKLMNPQRSTVWY"
    s = list(seq)
    idxs = rng.sample(range(len(s)), k=min(n_mut, len(s)))
    for i in idxs:
        choices = [c for c in aa if c != s[i]]
        s[i] = rng.choice(choices)
    return "".join(s)

def make_dates(rng: random.Random, n: int, start=date(2018, 1, 1), end=date(2025, 12, 31)):
    delta_days = (end - start).days
    return [start + timedelta(days=rng.randint(0, delta_days)) for _ in range(n)]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def generate_biosieve_dataset(
    n: int = 1000,
    seed: int = 13,
    n_clusters: int = 80,
    n_groups: int = 40,
    exact_dup_rate: float = 0.08,
    near_dup_rate: float = 0.25,
    target_name: str = "target",
) -> pd.DataFrame:
    """
    Generates a synthetic dataset with:
    - exact duplicates (same sequence)
    - near-duplicates (mutations within cluster)
    - group labels (e.g., study/patient/batch)
    - cluster_id (homology/embedding/structure cluster)
    - temporal column (date)
    - continuous regression target (column `target_name`)
    """
    rng = random.Random(seed)

    sources = ["uniprot", "literature", "synthetic", "internal_db"]
    species = [
        "Homo_sapiens",
        "Mus_musculus",
        "Rattus_norvegicus",
        "Bos_taurus",
        "Escherichia_coli",
        "Unknown",
    ]

    cluster_ids = [f"clust_{i:03d}" for i in range(1, n_clusters + 1)]
    group_ids = [f"study_{chr(65 + (i % 26))}{i//26:02d}" for i in range(n_groups)]

    # Cluster prototypes + a latent cluster-level target mean
    cluster_proto = {}
    cluster_target_mu = {}
    for cid in cluster_ids:
        length = rng.randint(18, 40)
        cluster_proto[cid] = random_peptide(rng, length)

        # cluster-specific mean in a plausible range (e.g., 0..1 or 0..100)
        # Here: map cluster index to a smooth pattern + randomness
        cidx = int(cid.split("_")[1])
        base = 0.3 + 0.25 * math.sin(cidx / 7.0) + 0.15 * (cidx % 5) / 5.0
        jitter = rng.uniform(-0.10, 0.10)
        cluster_target_mu[cid] = clamp(base + jitter, 0.05, 0.95)

    n_exact_dups = int(round(n * exact_dup_rate))
    n_near_dups = int(round(n * near_dup_rate))
    n_base = n - n_exact_dups - n_near_dups
    if n_base < 1:
        raise ValueError("Rates too high; n_base became < 1.")

    rows = []
    dates = make_dates(rng, n)

    for i in range(n_base):
        cid = rng.choice(cluster_ids)
        proto = cluster_proto[cid]

        seq = mutate_sequence(rng, proto, n_mut=rng.randint(0, 2))
        gid = rng.choice(group_ids)
        src = rng.choice(sources)
        sp = rng.choice(species)

        # structure_id: some are None, some shared within cluster
        structure_id = f"PDB_{rng.randint(1000, 9999)}" if rng.random() < 0.65 else "None"
        if rng.random() < 0.35:
            structure_id = f"PDB_{int(cid.split('_')[1]):04d}"

        embedding_id = f"emb_{i+1:04d}"

        # label: weakly correlated with cluster
        base_prob = 0.35 + (int(cid.split("_")[1]) % 7) * 0.05
        base_prob = min(0.85, max(0.15, base_prob))
        label = 1 if rng.random() < base_prob else 0

        # continuous target: cluster mean + length effect + gaussian noise
        # keep it in [0, 1] for simplicity (easy for demos)
        length_effect = clamp((len(seq) - 18) / (40 - 18), 0.0, 1.0)  # 0..1
        noise = rng.gauss(0.0, 0.07)
        target = clamp(0.65 * cluster_target_mu[cid] + 0.25 * length_effect + noise, 0.0, 1.0)

        rows.append(
            {
                "id": f"pep_{i+1:04d}",
                "sequence": seq,
                "label": label,
                "group": gid,
                "cluster_id": cid,
                "source": src,
                "species": sp,
                "structure_id": structure_id,
                "embedding_id": embedding_id,
                "date": str(dates[i]),
                target_name: float(target),
            }
        )

    # Near duplicates: mutate around an existing row; target stays close but not identical
    for _ in range(n_near_dups):
        base = rng.choice(rows)
        base_sequence = str(base["sequence"])
        base_label = int(base["label"])
        seq = mutate_sequence(rng, base_sequence, n_mut=rng.randint(1, 3))
        i = len(rows) + 1

        # target: near-dup => close but noisy (measurement variability)
        target = clamp(float(base[target_name]) + rng.gauss(0.0, 0.04), 0.0, 1.0)

        rows.append(
            {
                "id": f"pep_{i:04d}",
                "sequence": seq,
                "label": base_label if rng.random() < 0.75 else (1 - base_label),
                "group": base["group"] if rng.random() < 0.6 else rng.choice(group_ids),
                "cluster_id": base["cluster_id"],
                "source": base["source"],
                "species": base["species"],
                "structure_id": base["structure_id"] if rng.random() < 0.8 else "None",
                "embedding_id": f"emb_{i:04d}",
                "date": str(rng.choice(dates)),
                target_name: float(target),
            }
        )

    # Exact duplicates: identical sequence; target often identical, sometimes slight perturbation
    for _ in range(n_exact_dups):
        base = rng.choice(rows)
        base_label = int(base["label"])
        i = len(rows) + 1

        # with some probability, keep exactly same target; else tiny drift
        if rng.random() < 0.7:
            target = float(base[target_name])
        else:
            target = clamp(float(base[target_name]) + rng.gauss(0.0, 0.02), 0.0, 1.0)

        rows.append(
            {
                "id": f"pep_{i:04d}",
                "sequence": base["sequence"],
                "label": base_label if rng.random() < 0.6 else (1 - base_label),
                "group": base["group"] if rng.random() < 0.5 else rng.choice(group_ids),
                "cluster_id": base["cluster_id"],
                "source": base["source"],
                "species": base["species"],
                "structure_id": base["structure_id"],
                "embedding_id": f"emb_{i:04d}",
                "date": str(rng.choice(dates)),
                target_name: float(target),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_biosieve_dataset(n=1000, seed=13, target_name="target")
    df.to_csv("biosieve_example_dataset_1000.csv", index=False)
    print(df.head(10))
    print("Saved: biosieve_example_dataset_1000.csv", "rows:", len(df), "cols:", len(df.columns))
