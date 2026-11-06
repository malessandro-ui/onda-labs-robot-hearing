"""Tiny install-check and example CLI."""
import argparse
from . import MicrophoneArray


def main() -> None:
    parser = argparse.ArgumentParser(prog="onda-hearing")
    parser.add_argument("command", choices=("doctor", "version"))
    args = parser.parse_args()
    if args.command == "version":
        from importlib.metadata import version
        print(version("onda-robot-hearing"))
    else:
        import numpy as np
        a = MicrophoneArray(np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]]))
        a.validate_azimuth()
        print(f"OK: numpy={np.__version__}; array_channels={a.channels}; geometry_rank={a.geometry_rank}")
        print("Optional simulators and proprietary perception backend not installed or checked by this command.")
