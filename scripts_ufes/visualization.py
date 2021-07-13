import argparse
import os
import re
from pathlib import Path

import pandas as pd
import json
import matplotlib.pyplot as plt


def lastmatch_file(log, match_str):
    lastmatch = None
    for line in log:
        if match_str in line:
            lastmatch = line

    return lastmatch


def load_dict(line):
    json_string = re.findall('{.*}', line)[0].replace("'", "\"")
    obj = json.loads(json_string)

    return obj


def read_actions(log):
    model_actions_match = "INFO:ml.rl.training.loss_reporter:The distribution of model actions :"
    logged_actions_match = "INFO:ml.rl.training.loss_reporter:The distribution of logged actions :"

    file = open(log, 'r')
    file = file.readlines()

    model_actions = load_dict(lastmatch_file(file, model_actions_match))
    logged_actions = load_dict(lastmatch_file(file, logged_actions_match))

    return logged_actions, model_actions


def load_model(path):
    logged_act, model_act = read_actions(path)

    ds = [logged_act, model_act]
    data = {}
    for k in logged_act.keys():
        data[k] = list(ds_elem[k] for ds_elem in ds)

    df = pd.DataFrame.from_dict(data, orient="index", columns=["Logged", "Model"])

    return df


def actions_comparison_bar(path, output="/tmp", use_pkl=False):
    if isinstance(path, list):
        # Multiplos modelos
        df_list = []
        for num, dirname in enumerate(path):
            if use_pkl:
                tmp_df = pd.read_pickle(dirname)
            else:
                tmp_df = load_model(dirname)
            tmp_df.rename(columns={'Model': F'Learned policy (seed {num+1})'}, inplace=True)
            df_list.append(tmp_df)
        df = pd.concat(df_list, axis="columns")

        df_logged = df["Logged"].mean(axis="columns").copy()
        df = df.drop(columns="Logged").copy()

        df["Logged policy (mean)"] = df_logged
    else:
        # Único modelo
        if use_pkl:
            df = pd.read_pickle(path)
        else:
            df = load_model(path)

    ax = df.plot.bar(rot=0, title="Recommended Actions")
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")

    df.to_pickle(F"{output}/actions_comparison_bar.pkl")
    plt.savefig(F"{output}/actions_comparison_bar.png")


##
# Example: python visualization.py --path /tmp/docs/kmeans/result --output /tmp/docs/kmeans --plot_all
# Example: python visualization.py --path /tmp/docs/kmeans/result/1 --output /tmp/docs/kmeans/1
##
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="main path for policy actions file", default=os.getcwd())
    parser.add_argument("--subpath", type=str, help="subpath path for policy actions file", default="outputs")
    parser.add_argument("--file", type=str, help="logfile for policy actions", default="policy_training.log")
    parser.add_argument("--output", type=str, help="output path")
    parser.add_argument("--plot_all", help="plot all models together path", action="store_true")
    parser.add_argument("--use_pkl", help="use a pickle file of the dataframe instead of logfile", action="store_true")

    args = parser.parse_args()

    logpath = Path(args.path, args.subpath, args.file)

    if args.output:
        path_output = Path(args.output)
    else:
        path_output = Path(args.path, args.subpath)

    if args.plot_all:
        main_path = Path(args.path)
        (_, dirnames, _) = next(os.walk(main_path))
        path_list = [Path(args.path, dname, args.subpath, args.file) for dname in dirnames]
        actions_comparison_bar(path_list, output=path_output, use_pkl=args.use_pkl)
    else:
        print("")
        actions_comparison_bar(logpath, output=path_output, use_pkl=args.use_pkl)
