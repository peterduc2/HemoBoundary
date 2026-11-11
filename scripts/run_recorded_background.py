"""Run the recorded-background evaluation using the unchanged model source."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.recorded_background import main
if __name__ == '__main__':
    main()
