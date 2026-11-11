"""Run the participant-grouped nested classical comparison."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nested_classical import main
if __name__ == '__main__':
    main()
