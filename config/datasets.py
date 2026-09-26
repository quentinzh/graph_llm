"""Dataset path resolution for graph_llm (PLEASER categories only)."""

from __future__ import annotations

from pathlib import Path

from graph_llm.dataload.pleaser import (
    PLEASER_DATASET_NAMES,
    canonical_pleaser_name,
    pleaser_dataset_dir,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent


def resolve_dataset_paths(args) -> None:
    """Resolve args.dataset_name and args.data_dir for a PLEASER dataset.

    在仓库 ``data/{Name}/`` 或 ``args.data_dir/{Name}/`` 下查找
    ``sequences.jsonl`` / ``reviews.jsonl`` / ``items.jsonl``。
    """
    user_name = str(args.dataset_name).strip().strip("/")
    canonical = canonical_pleaser_name(user_name)
    if canonical is None:
        raise ValueError(
            f"Unknown dataset {user_name!r}. "
            f"Supported PLEASER datasets: {', '.join(PLEASER_DATASET_NAMES)}"
        )

    search_roots: list[Path] = []
    for root in (Path(args.data_dir), REPO_ROOT / "data", PACKAGE_ROOT / "data"):
        root = root.resolve()
        if root not in search_roots:
            search_roots.append(root)

    tried: list[str] = []
    for root in search_roots:
        dataset_dir = root / canonical
        tried.append(str(dataset_dir))
        try:
            pleaser_dataset_dir(root, canonical)
        except (FileNotFoundError, ValueError):
            continue
        args.dataset_name = canonical
        args.data_dir = str(root)
        print(
            f"Resolved dataset '{user_name}' -> "
            f"dataset_name={args.dataset_name!r}, data_dir={root}"
        )
        return

    raise FileNotFoundError(
        f"Could not resolve PLEASER dataset {user_name!r} ({canonical}). "
        f"Tried directories:\n  " + "\n  ".join(tried)
    )
