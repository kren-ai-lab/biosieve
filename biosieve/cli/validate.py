from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import pandas as pd
import typer

from biosieve.cli.common import LOG_FILE_OPTION, LOG_LEVEL_OPTION, QUIET_OPTION, setup_runtime
from biosieve.types import Columns

if TYPE_CHECKING:
    from biosieve.core.registry import StrategyRegistry

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

INPUT_DATA_OPTION = typer.Option(
    ...,
    "--input-data",
    "-i",
    help="Input dataset CSV path.",
)
COLUMNS_OPTION = typer.Option(
    None,
    "--columns",
    help="Optional columns JSON string or path to JSON. Defaults to id='id', sequence='sequence'.",
)
EMBEDDINGS_OPTION = typer.Option(
    None,
    "--embeddings",
    help="Embeddings .npy path (optional).",
)
EMBEDDING_IDS_OPTION = typer.Option(
    None,
    "--embedding-ids",
    help="Embedding ids CSV path (optional).",
)
EMBEDDING_IDS_COLUMN_OPTION = typer.Option(
    "id",
    "--embedding-ids-column",
    help="Column name for ids in embedding ids CSV.",
    show_default=True,
)
DESCRIPTORS_PREFIX_OPTION = typer.Option(
    "desc_",
    "--descriptors-prefix",
    help="Prefix for descriptor columns check.",
    show_default=True,
)
EDGES_OPTION = typer.Option(
    None,
    "--edges",
    help="Structural edges CSV path (optional).",
)
EDGES_ID1_COLUMN_OPTION = typer.Option(
    "id1",
    "--edges-id1-column",
    help="Edges CSV id1 column.",
    show_default=True,
)
EDGES_ID2_COLUMN_OPTION = typer.Option(
    "id2",
    "--edges-id2-column",
    help="Edges CSV id2 column.",
    show_default=True,
)
EDGES_VALUE_COLUMN_OPTION = typer.Option(
    "distance",
    "--edges-value-column",
    help="Edges CSV value column.",
    show_default=True,
)
MMSEQS2_BINARY_OPTION = typer.Option(
    "mmseqs",
    "--mmseqs2-binary",
    help="Name/path to mmseqs binary to check.",
    show_default=True,
)
STRATEGY_OPTION = typer.Option(
    None,
    "--strategy",
    "-s",
    help="Optional strategy name to validate required artefacts for.",
)
KIND_OPTION = typer.Option(
    "reduce",
    "--kind",
    help="Whether --strategy refers to a reducer or a splitter.",
    show_default=True,
)
FAIL_FAST_OPTION = typer.Option(
    False,
    "--fail-fast/--no-fail-fast",
    help="Stop at first error.",
    show_default=True,
)
REPORT_OUTPUT_OPTION = typer.Option(
    None,
    "--report-output",
    help="Optional JSON report output path.",
)


# -----------------------------
# Utilities
# -----------------------------
def _load_cols(cols_arg: str | None) -> Columns:
    # Keep this intentionally simple: default is enough for now
    if not cols_arg:
        return Columns(id_col="id", seq_col="sequence")

    p = Path(cols_arg)
    if p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        text = cols_arg

    # Support JSON-like dict string only (fast path). YAML parsing can be added later.
    import json

    d = json.loads(text)
    return Columns(**d)


def _jsonable(x: object) -> JSONValue:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        return _jsonable(asdict(x))
    return str(x)


def _check_exists(path: str | None, label: str) -> tuple[bool, str]:
    if not path:
        return True, f"SKIP {label}: not provided"
    p = Path(path)
    if not p.exists():
        return False, f"FAIL {label}: missing path '{path}'"
    return True, f"OK   {label}: found '{path}'"


def _check_unique_ids(df: pd.DataFrame, id_col: str) -> tuple[bool, str]:
    if id_col not in df.columns:
        return False, f"FAIL dataset: missing id column '{id_col}'"
    n = len(df)
    u = df[id_col].astype(str).nunique()
    if u != n:
        return False, f"FAIL dataset: ids not unique ({u} unique for {n} rows)"
    return True, f"OK   dataset: ids unique ({n} rows)"


def _check_seq_col(df: pd.DataFrame, seq_col: str | None) -> tuple[bool, str]:
    if not seq_col:
        return True, "SKIP dataset: seq_col not configured"
    if seq_col not in df.columns:
        return False, f"FAIL dataset: missing sequence column '{seq_col}'"
    # allow empty sequences in validate? safer to flag
    empty = (df[seq_col].astype(str).str.len() == 0).sum()
    if empty > 0:
        return False, f"FAIL dataset: {empty} empty sequences in '{seq_col}'"
    return True, f"OK   dataset: sequence column present ('{seq_col}')"


def _check_embeddings_alignment(
    df: pd.DataFrame,
    id_col: str,
    embeddings_path: str | None,
    ids_path: str | None,
    ids_col: str,
) -> tuple[bool, str]:
    if not embeddings_path and not ids_path:
        return True, "SKIP embeddings: not provided"
    if not embeddings_path or not ids_path:
        return False, "FAIL embeddings: must provide BOTH --embeddings and --embedding-ids"

    pX = Path(embeddings_path)
    pI = Path(ids_path)
    if not pX.exists() or not pI.exists():
        missing = []
        if not pX.exists():
            missing.append(str(pX))
        if not pI.exists():
            missing.append(str(pI))
        return False, f"FAIL embeddings: missing file(s): {missing}"

    # Load ids CSV
    ids_df = pd.read_csv(pI)
    if ids_col not in ids_df.columns and len(ids_df.columns) == 1:
        # tolerate single-col CSV
        ids_col_eff = ids_df.columns[0]
    else:
        ids_col_eff = ids_col
    if ids_col_eff not in ids_df.columns:
        return False, f"FAIL embeddings: ids column '{ids_col}' not found in {list(ids_df.columns)}"

    emb_ids = ids_df[ids_col_eff].astype(str).tolist()

    # Load embeddings array shape only
    X = np.load(pX, mmap_mode="r")
    if X.ndim != 2:
        return False, f"FAIL embeddings: expected 2D array (N,D), got shape {tuple(X.shape)}"
    if len(emb_ids) != X.shape[0]:
        return False, f"FAIL embeddings: ids length {len(emb_ids)} != embeddings rows {X.shape[0]}"

    # Coverage check relative to dataset
    ds_ids = set(df[id_col].astype(str).tolist())
    emb_set = set(emb_ids)
    present = len([x for x in ds_ids if x in emb_set])
    coverage = present / len(ds_ids) if ds_ids else 0.0

    if present == 0:
        return False, "FAIL embeddings: none of dataset ids are present in embedding ids"
    return (
        True,
        f"OK   embeddings: aligned (rows={X.shape[0]}, dim={X.shape[1]}), dataset coverage={coverage:.3f}",
    )


def _check_descriptors(df: pd.DataFrame, prefix: str) -> tuple[bool, str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return True, f"SKIP descriptors: no columns with prefix '{prefix}'"
    sub = df[cols]
    # numeric coercion check
    coerced = sub.apply(pd.to_numeric, errors="coerce")
    n_nans = int(coerced.isna().sum().sum())
    if n_nans > 0:
        return (
            False,
            f"FAIL descriptors: non-numeric/NaN values detected (total NaNs after coercion: {n_nans})",
        )
    return True, f"OK   descriptors: {len(cols)} numeric columns with prefix '{prefix}'"


def _check_edges(
    df: pd.DataFrame,
    id_col: str,
    edges_path: str | None,
    id1_col: str,
    id2_col: str,
    value_col: str,
) -> tuple[bool, str]:
    if not edges_path:
        return True, "SKIP edges: not provided"
    p = Path(edges_path)
    if not p.exists():
        return False, f"FAIL edges: missing path '{edges_path}'"

    e = pd.read_csv(p)
    for c in (id1_col, id2_col, value_col):
        if c not in e.columns:
            return False, f"FAIL edges: missing column '{c}' in edges CSV. Found: {list(e.columns)}"

    # ids coverage
    ds_ids = set(df[id_col].astype(str).tolist())
    id1 = e[id1_col].astype(str)
    id2 = e[id2_col].astype(str)
    present_any = int(((id1.isin(ds_ids)) | (id2.isin(ds_ids))).sum())
    if present_any == 0:
        return False, "FAIL edges: no edges connect to dataset ids (check id space)"
    return True, f"OK   edges: loaded {len(e)} edges; connects to dataset ids (at least {present_any} rows)"


def _check_mmseqs2(binary: str) -> tuple[bool, str]:
    import shutil

    path = shutil.which(binary)
    if path is None:
        return False, f"FAIL mmseqs2: binary '{binary}' not found in PATH"
    return True, f"OK   mmseqs2: found '{path}'"


def _strategy_requires(strategy: str, kind: str) -> dict[str, bool]:
    """Minimal requirements mapping.
    Adjust as you add more strategies.
    """
    if kind == "reduce":
        if strategy == "embedding_cosine":
            return {"embeddings": True}
        if strategy == "descriptor_euclidean":
            return {"descriptors": True}
        if strategy == "structural_distance":
            return {"edges": True}
        if strategy == "mmseqs2":
            return {"mmseqs2": True}
        if strategy in {"kmer_jaccard", "identity_greedy", "exact"}:
            return {"sequence": True}
    if kind == "split":
        if strategy in {"stratified"}:
            return {"label": True}
        if strategy in {"stratified_numeric", "stratified_numeric_kfold"}:
            return {"label": True}
        if strategy in {"group", "group_kfold"}:
            return {"group": True}
        if strategy in {"time"}:
            return {"date": True}
        if strategy in {"distance_aware", "distance_aware_kfold"}:
            return {"embeddings_or_descriptors": True}
        if strategy in {"homology_aware"}:
            return {"mmseqs2": True}
        if strategy in {"cluster_aware"}:
            return {"cluster": True}
    return {}


# -----------------------------
# Main validate runner
# -----------------------------
def validate(
    input_data: Path = INPUT_DATA_OPTION,
    columns: str | None = COLUMNS_OPTION,
    embeddings: Path | None = EMBEDDINGS_OPTION,
    embedding_ids: Path | None = EMBEDDING_IDS_OPTION,
    embedding_ids_column: str = EMBEDDING_IDS_COLUMN_OPTION,
    descriptors_prefix: str = DESCRIPTORS_PREFIX_OPTION,
    edges: Path | None = EDGES_OPTION,
    edges_id1_column: str = EDGES_ID1_COLUMN_OPTION,
    edges_id2_column: str = EDGES_ID2_COLUMN_OPTION,
    edges_value_column: str = EDGES_VALUE_COLUMN_OPTION,
    mmseqs2_binary: str = MMSEQS2_BINARY_OPTION,
    strategy: str | None = STRATEGY_OPTION,
    kind: str = KIND_OPTION,
    fail_fast: bool = FAIL_FAST_OPTION,
    report_output: Path | None = REPORT_OUTPUT_OPTION,
    log_level: str = LOG_LEVEL_OPTION,
    quiet: bool = QUIET_OPTION,
    log_file: Path | None = LOG_FILE_OPTION,
) -> None:
    """Validate dataset inputs and optional artefacts before split/reduce runs."""
    if kind not in {"reduce", "split"}:
        raise typer.BadParameter("kind must be one of: reduce, split")

    registry = setup_runtime(log_level, quiet, log_file)
    args = SimpleNamespace(
        in_path=str(input_data),
        cols=columns,
        embeddings=str(embeddings) if embeddings is not None else None,
        embedding_ids=str(embedding_ids) if embedding_ids is not None else None,
        embedding_ids_col=embedding_ids_column,
        descriptors_prefix=descriptors_prefix,
        edges=str(edges) if edges is not None else None,
        edges_id1_col=edges_id1_column,
        edges_id2_col=edges_id2_column,
        edges_value_col=edges_value_column,
        mmseqs2_binary=mmseqs2_binary,
        strategy=strategy,
        kind=kind,
        fail_fast=fail_fast,
        report=str(report_output) if report_output is not None else None,
    )
    _run_validate(args, registry)


def _run_validate(args: SimpleNamespace, registry: StrategyRegistry) -> None:
    cols = _load_cols(args.cols)

    results: list[dict[str, JSONValue]] = []
    errors = 0

    def record(ok: bool, msg: str) -> None:
        nonlocal errors
        results.append({"ok": bool(ok), "message": msg})
        print(msg)
        if not ok:
            errors += 1
            if args.fail_fast:
                raise SystemExit(1)

    # --- dataset read ---
    ok, msg = _check_exists(args.in_path, "dataset")
    record(ok, msg)
    df = pd.read_csv(args.in_path)

    # base checks
    ok, msg = _check_unique_ids(df, cols.id_col)
    record(ok, msg)

    # optional checks (sequence exists)
    ok, msg = _check_seq_col(df, cols.seq_col)
    record(ok, msg)

    # embeddings alignment
    ok, msg = _check_embeddings_alignment(
        df,
        cols.id_col,
        args.embeddings,
        args.embedding_ids,
        args.embedding_ids_col,
    )
    record(ok, msg)

    # descriptor sanity
    ok, msg = _check_descriptors(df, args.descriptors_prefix)
    record(ok, msg)

    # structural edges
    ok, msg = _check_edges(
        df,
        cols.id_col,
        args.edges,
        args.edges_id1_col,
        args.edges_id2_col,
        args.edges_value_col,
    )
    record(ok, msg)

    # mmseqs2
    ok, msg = _check_mmseqs2(args.mmseqs2_binary)
    # only fail if user is validating mmseqs2 explicitly or strategy requires it
    must_mmseqs = False
    if args.strategy and _strategy_requires(args.strategy, args.kind).get("mmseqs2", False):
        must_mmseqs = True
    if must_mmseqs:
        record(ok, msg)
    else:
        # make it informational
        results.append({"ok": bool(ok), "message": msg, "informational": True})
        print(msg)

    # Strategy-aware requirements
    if args.strategy:
        # validate that strategy exists in registry light/full
        if args.kind == "reduce":
            if not registry.has_reducer(args.strategy):
                raise ValueError(
                    f"Unknown reducer strategy '{args.strategy}'. "
                    f"Available: {sorted(registry.list_reducers())}"
                )
        elif not registry.has_splitter(args.strategy):
            raise ValueError(
                f"Unknown splitter strategy '{args.strategy}'. "
                f"Available: {sorted(registry.list_splitters())}"
            )

        req = _strategy_requires(args.strategy, args.kind)

        # resolve requirements against provided artefacts and cols
        if req.get("sequence", False):
            ok, msg = _check_seq_col(df, cols.seq_col)
            record(ok, f"[strategy={args.strategy}] {msg}")

        if req.get("descriptors", False):
            ok, msg = _check_descriptors(df, args.descriptors_prefix)
            record(ok, f"[strategy={args.strategy}] {msg}")

        if req.get("embeddings", False):
            ok, msg = _check_embeddings_alignment(
                df, cols.id_col, args.embeddings, args.embedding_ids, args.embedding_ids_col
            )
            record(ok, f"[strategy={args.strategy}] {msg}")

        if req.get("edges", False):
            ok, msg = _check_edges(
                df, cols.id_col, args.edges, args.edges_id1_col, args.edges_id2_col, args.edges_value_col
            )
            record(ok, f"[strategy={args.strategy}] {msg}")

        if req.get("label", False):
            if not cols.label_col:
                record(False, f"[strategy={args.strategy}] FAIL dataset: cols.label_col not configured")
            elif cols.label_col not in df.columns:
                record(
                    False, f"[strategy={args.strategy}] FAIL dataset: missing label column '{cols.label_col}'"
                )
            else:
                record(
                    True, f"[strategy={args.strategy}] OK   dataset: label column '{cols.label_col}' present"
                )

        if req.get("group", False):
            if not cols.group_col:
                record(False, f"[strategy={args.strategy}] FAIL dataset: cols.group_col not configured")
            elif cols.group_col not in df.columns:
                record(
                    False, f"[strategy={args.strategy}] FAIL dataset: missing group column '{cols.group_col}'"
                )
            else:
                record(
                    True, f"[strategy={args.strategy}] OK   dataset: group column '{cols.group_col}' present"
                )

        if req.get("cluster", False):
            if not cols.cluster_col:
                record(False, f"[strategy={args.strategy}] FAIL dataset: cols.cluster_col not configured")
            elif cols.cluster_col not in df.columns:
                record(
                    False,
                    f"[strategy={args.strategy}] FAIL dataset: missing cluster column '{cols.cluster_col}'",
                )
            else:
                record(
                    True,
                    f"[strategy={args.strategy}] OK   dataset: cluster column '{cols.cluster_col}' present",
                )

        if req.get("date", False):
            if not cols.date_col:
                record(False, f"[strategy={args.strategy}] FAIL dataset: cols.date_col not configured")
            elif cols.date_col not in df.columns:
                record(
                    False, f"[strategy={args.strategy}] FAIL dataset: missing date column '{cols.date_col}'"
                )
            else:
                record(
                    True, f"[strategy={args.strategy}] OK   dataset: date column '{cols.date_col}' present"
                )

        if req.get("embeddings_or_descriptors", False):
            ok_emb = args.embeddings is not None and args.embedding_ids is not None
            ok_desc = any(c.startswith(args.descriptors_prefix) for c in df.columns)
            if not (ok_emb or ok_desc):
                record(
                    False,
                    f"[strategy={args.strategy}] FAIL need embeddings "
                    f"(--embeddings + --embedding-ids) OR descriptor columns "
                    f"(prefix '{args.descriptors_prefix}')",
                )
            else:
                record(True, f"[strategy={args.strategy}] OK   embeddings/descriptors requirement satisfied")

    # Optional report output
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "0.1",
            "in_path": args.in_path,
            "columns": _jsonable(cols),
            "strategy": args.strategy,
            "kind": args.kind,
            "results": results,
            "n_errors": int(errors),
        }
        import json

        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if errors > 0:
        raise SystemExit(1)
