"""Dataset path resolution for graph_llm."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent


SEQUENTIAL_MARKERS = (
    "sequences.jsonl",
    "reviews.jsonl",
    "items.jsonl",
    "split.json",
)


def is_sequential_dataset_dir(path: Path) -> bool:
    return all((path / name).is_file() for name in SEQUENTIAL_MARKERS)


def dataset_name_candidates(name: str) -> list[str]:
    """Return canonical dataset name candidates for a user-provided short name."""
    clean = str(name).strip().strip("/")
    if not clean:
        return []
    candidates = [clean]
    if not clean.startswith("Amazon/"):
        candidates.append(f"Amazon/{clean}")
    return list(dict.fromkeys(candidates))


def resolve_dataset_paths(args) -> None:
    """Resolve args.dataset_name and args.data_dir from user input.

    Sequential PLEASER datasets: sequences/reviews/items/split.jsonl+json.
    Legacy explain datasets: reviews.pickle under Amazon/... or flat names.
    """
    user_name = str(args.dataset_name).strip().strip("/")
    candidates = dataset_name_candidates(user_name)
    if not candidates:
        raise ValueError("dataset_name must not be empty")

    search_roots = []
    for root in [Path(args.data_dir), PACKAGE_ROOT / "data", REPO_ROOT / "data"]:
        root = root.resolve()
        if root not in search_roots:
            search_roots.append(root)

    tried = []
    for root in search_roots:
        for canonical in candidates:
            seq_dir = root / canonical
            tried.append(str(seq_dir))
            if is_sequential_dataset_dir(seq_dir):
                args.dataset_name = canonical
                args.data_dir = str(root)
                args.dataset_format = "sequential"
                print(
                    f"Resolved sequential dataset '{user_name}' -> "
                    f"dataset_name={args.dataset_name!r}, data_dir={root}"
                )
                return
            reviews_path = seq_dir / "reviews.pickle"
            tried.append(str(reviews_path))
            if reviews_path.is_file():
                if "/" in canonical:
                    args.dataset_name = canonical.rstrip("/") + "/"
                else:
                    args.dataset_name = canonical
                args.data_dir = str(root)
                args.dataset_format = "legacy"
                print(
                    f"Resolved legacy dataset '{user_name}' -> "
                    f"dataset_name={args.dataset_name!r}, data_dir={root}"
                )
                return

    raise FileNotFoundError(
        f"Could not resolve dataset {user_name!r}. Tried:\n  "
        + "\n  ".join(tried)
    )
