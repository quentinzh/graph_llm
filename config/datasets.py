"""Dataset path resolution for graph_llm."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent


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

    Searches args.data_dir (default graph_llm/data) then repo data/ for
    reviews.pickle under each canonical name candidate.
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
            reviews_path = root / canonical / "reviews.pickle"
            tried.append(str(reviews_path))
            if reviews_path.is_file():
                args.dataset_name = canonical
                args.data_dir = str(root)
                print(
                    f"Resolved dataset '{user_name}' -> "
                    f"dataset_name={canonical!r}, data_dir={root}"
                )
                return

    raise FileNotFoundError(
        f"Could not resolve dataset {user_name!r}. Tried:\n  "
        + "\n  ".join(tried)
    )
