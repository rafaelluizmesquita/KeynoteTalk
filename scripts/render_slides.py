"""Render the continuous short version and its eight slide clips."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sbgea_keynote.visualization.render_stages import main  # noqa: E402

if __name__ == "__main__":
    main()

