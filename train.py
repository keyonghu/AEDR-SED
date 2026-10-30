"""Train AEDR-SED on the dataset selected in config.py."""

import argparse
import sys

import config


def main(evaluate=False):
    description = "Evaluate an AEDR-SED checkpoint." if evaluate else __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset", choices=["audioset", "desed"], default=config.DATASET
    )
    parser.add_argument("--checkpoint", help="Trained checkpoint to evaluate.")
    args, forwarded = parser.parse_known_args()
    if args.checkpoint:
        if not evaluate:
            parser.error("Use --resume_checkpoint to resume training.")
        forwarded += ["--lightning_checkpoint", args.checkpoint]
    if evaluate:
        forwarded.append("--evaluate")
        if not any(
            value == "--pretrained" or value.startswith("--pretrained=")
            for value in forwarded
        ):
            forwarded += ["--pretrained", "scratch"]
    if args.dataset == "audioset":
        from models.engine import main as run
    else:
        from models.desed import main as run
    previous_arguments = sys.argv
    try:
        sys.argv = [previous_arguments[0], *forwarded]
        run()
    finally:
        sys.argv = previous_arguments


if __name__ == "__main__":
    main()
