"""Run exploratory recorded-background transfer diagnostics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.transfer_diagnostics import main
if __name__ == '__main__':
    main()
