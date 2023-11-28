import sys
from . import train_stage, infer_stage, eval_stage


def main():
    if len(sys.argv) < 2:
        print("Usage: acorn {train|infer|eval}")
        return

    command = sys.argv[1]
    if command == "train":
        train_stage.main()
    elif command == "infer":
        infer_stage.main()
    elif command == "eval":
        eval_stage.main()
    else:
        print(f"Unknown command: {command}")
