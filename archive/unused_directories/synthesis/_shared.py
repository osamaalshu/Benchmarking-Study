from pathlib import Path
from typing import Optional


def resolve_upstream_repo_path(ext_repo: Optional[str]) -> Path:
    if ext_repo:
        p = Path(ext_repo).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Upstream repo not found: {p}")
        return p
    # default location inside this project
    here = Path(__file__).resolve().parents[0]
    default = here / "external"
    if not default.exists():
        raise FileNotFoundError(
            f"Upstream repo not found at {default}. Run: python synthesis/setup_external.py --clone or pass --ext_repo"
        )
    return default


def default_results_dirs(name: str):
    root = Path(__file__).resolve().parents[0]
    checkpoints = root / "checkpoints" / name
    results = root / "results" / name
    checkpoints.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return checkpoints, results


