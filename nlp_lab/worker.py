import sys
import traceback
from pathlib import Path

from .data import save_json
from .training import evaluate_run, execute_run


if __name__ == "__main__":
    run, mode, split = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    try:
        if mode == "train":
            execute_run(run)
        else:
            evaluate_run(run, split)
            save_json(run / "evaluation_status.json", {"status": "completed", "split": split})
    except Exception as e:
        if mode != "train":
            save_json(run / "evaluation_status.json", {"status": "failed", "split": split, "error": str(e)})
        traceback.print_exc()
        sys.exit(1)
