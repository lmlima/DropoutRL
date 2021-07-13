import argparse
import random
import sys

import torch
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, nargs='+', help="Command to run")
    parser.add_argument("--seed", type=int, help="Seed of this run", default=1)

    args = parser.parse_args()

    seed = args.seed

    command_argv = args.command[0].split()
    runfile = command_argv[0]
    sys.argv = command_argv

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    exec(open(runfile).read())

