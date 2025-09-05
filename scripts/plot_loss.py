#!/usr/bin/env python3
"""
Plot training loss curves from checkpoint CSV logs.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    plt.figure()
    for root, _, files in os.walk(args.log_dir):
        for f in files:
            if f.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, f))
                if "loss" in df.columns:
                    plt.plot(df["step"], df["loss"], label=os.path.basename(root))

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
