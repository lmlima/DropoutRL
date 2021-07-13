import argparse
import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

###
# Extract CSV of Tensorflow summaries
# https://stackoverflow.com/a/52095336/11702735
###

FOLDER_NAME = 'csv'

METHOD_DICT = {
    "CPE_Reward_Weighted_Sequential_Doubly_Robust": "SWDR",
    "CPE_Reward_MAGIC": "MAGIC",
}


def tabulate_events(dpath, subpath, tags):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname, subpath)).Reload()
                         for dname in os.listdir(dpath) if dname != FOLDER_NAME]

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath, subpath, input_tags):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath, subpath, input_tags)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag), index_label="step")


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


def plot_csv(dpath):
    folder_path = os.path.join(dpath, 'csv')

    dfs = []
    for filename in glob.glob(F"{folder_path}/*.csv"):
        method = Path(filename).stem

        df = pd.read_csv(filename)
        df.rename(columns={'step': 'Step'}, inplace=True)
        df.set_index('Step', inplace=True)
        df = df.stack().reset_index().rename(columns={'level_1': 'run', 0: 'Value'})

        df["Method"] = METHOD_DICT[method] if method in METHOD_DICT.keys() else method
        dfs.append(df)

    dados = pd.concat(dfs)

    ax = sns.lineplot(x="Step", y="Value", hue="Method", data=dados)
    ax.axhline(1, ls='--', c='red', alpha=0.5)

    fig_filename = F"{folder_path}/CPE_Reward.png"
    plt.savefig(fig_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--subpath", type=str, help="subpath path for tensorboard files", default="outputs")
    parser.add_argument("--tags", type=str, help="Tags to use",
                        default=['CPE/Reward/Weighted_Sequential_Doubly_Robust', 'CPE/Reward/MAGIC'])

    args = parser.parse_args()

    path = Path(args.path)

    to_csv(path, args.subpath, args.tags)
    plot_csv(path)
