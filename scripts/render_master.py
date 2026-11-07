"""Render the full mission keynote by default; pass renderer flags as needed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbgea_keynote.visualization.render_mission import main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--master")
    main()

