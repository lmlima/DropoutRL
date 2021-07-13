import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
import importlib

import pandas as pd
import numpy as np
import random

from typing import Any, Dict, Optional
from sklearn.cluster import KMeans, OPTICS
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)


class Students:
    def __init__(self, config_file, seed=0):

        with open(config_file) as json_file:
            self.config = json.load(json_file)

        # Define seed
        self.config["seed"] = str(seed)
        logger.info(F"Seed: {seed}")
        np.random.seed(seed)
        random.seed(seed)
        # torch.manual_seed(seed)

        self.dados = None
        self.extras = None

        self.ACTIONS = None
        self.REWARDS = None

        # Setup log
        if "output_path" in self.config:
            output_path = Path(F"{self.config['output_path']}", self.config["seed"])
            output_path.mkdir(parents=True, exist_ok=True)
            log_file = F"{output_path}/gen_discrete_student.log"
        else:
            log_file = "gen_discrete_student.log"
        filelogHandler = logging.FileHandler(filename=log_file)
        filelogHandler.setLevel(logging.DEBUG)
        logger.addHandler(filelogHandler)

        # Load reward class
        if "reward" in self.config:
            class_name = self.config["reward"]
            module_name = "evasao.Rewards"
        else:
            class_name = "SparsePReward"
            module_name = "evasao.Rewards"
        module = importlib.import_module(module_name)
        self.reward_class = getattr(module, class_name)

        self.STATES_columns = None
        self.ACTIONS_columns = None

        self.action_probability_data = {}

        self.ACT_PLANO_ESTUDO = None
        self.ACT_TIPO_AUXILIO = None
        self.ACTIONS_NAMES = None
        self.ACTIONS_INDEXES = None

        self.cluster_centers = None

    def init_data(self):
        # path_dados = "/home/leandro/Documentos/doutorado/dados/Ufes/dados1_anon_leandro-rl.pkl"
        # path_dados = "/tmp/Ufes/dados1_anon_leandro-rl.pkl"
        path_dados = self.config["data_path"]

        self.dados = pd.read_pickle(path_dados)

        # Filter data
        self.dados = self.dados[((self.dados.FORMA_EVASAO_last != "Sem evasÃ£o") |
                                 (self.dados.FORMA_EVASAO_last != "Outros"))]

        # Extra info dataframe
        self.extras = pd.DataFrame(index=self.dados.index)
        self.prepare_data_extras()

        # self.STATES_columns = list(set(self.dados.columns) - set(self.config["target"]).union(set(self.config["actions"])))
        # self.STATES_columns = list(set(self.dados.columns) - set(self.config["target"]).union(set(self.config["actions"])))
        self.STATES_columns = self.config["states"]
        self.ACTIONS_columns = self.config["actions"]

        # Gerar lista de ações
        PLANO_ESTUDO_list = self.dados["PLANO_ESTUDO_last"].unique().tolist()
        self.ACT_PLANO_ESTUDO = {PLANO_ESTUDO_list.index(elem): elem for elem in PLANO_ESTUDO_list}

        TIPO_AUXILIO_list = self.dados["TIPO_AUXILIO_last"].unique().tolist()
        self.ACT_TIPO_AUXILIO = {TIPO_AUXILIO_list.index(elem): elem for elem in TIPO_AUXILIO_list}

        # Combinacao de todas as possibilidades de (PLANO_ESTUDO, TIPO_AUXILIO)
        ACTIONS_LIST = list(itertools.product(PLANO_ESTUDO_list, TIPO_AUXILIO_list))

        # Cada ACTION = (PLANO_ESTUDO, TIPO_AUXILIO)
        self.ACTIONS_NAMES = {ACTIONS_LIST.index(elem): elem for elem in ACTIONS_LIST}
        logger.info(F"{self.ACTIONS_NAMES}\n")

        self.ACTIONS_INDEXES = {v: k for k, v in self.ACTIONS_NAMES.items()}

        self.prepare_actions()


        if ("discrete_state" in self.config) and ("use" in self.config["discrete_state"]) \
                and (self.config["discrete_state"]["use"] is True):
            self.discrete_state()


    def prepare_data_extras(self):
        self.prepare_last_semester_before_dropout()

    def prepare_last_semester_before_dropout(self):
        grp = self.dados.reset_index(level='PERIODO_DISCIPLINA').groupby('ID_CURSO_ALUNO')[
            ["PERIODO_DISCIPLINA", 'FORMA_EVASAO_last']]

        df = grp.tail(1).reset_index().set_index(['ID_CURSO_ALUNO', 'PERIODO_DISCIPLINA'])
        df_evasao = df[df.FORMA_EVASAO_last == "Desistência"]

        self.extras["last_semester_before_dropout"] = 0
        self.extras.loc[df_evasao.index, "last_semester_before_dropout"] = 1

    def prepare_actions(self):
        self.dados["action"] = self.dados[['PLANO_ESTUDO_last', 'TIPO_AUXILIO_last']] \
            .apply(tuple, axis="columns") \
            .map(self.ACTIONS_INDEXES.get) \
            .map(str)
        self.ACTIONS = self.dados["action"].copy()

    def discrete_state(self):
        """
            Discretiza o estado usando kmean
        """
        method = self.config["discrete_state"]["method"]
        logger.info(F"Clustering method: {method}")

        # Standardarizing data
        scaler = StandardScaler()
        # Fit on training set only.
        train_data = self.dados[self.config["states"]].fillna(0)
        scaler.fit(train_data)
        # Apply transform to both the training set and the test set.
        sample = scaler.transform(train_data)
        # test_img = scaler.transform(test_data)

        # Clustering state space
        if method == "xmeans":
            # sample = self.dados[self.config["states"]].fillna(0).values.tolist()
            initial_centers = None

            initializer = self.config["discrete_state"]["init"]
            if initializer == "k-means++":
                initial_centers = kmeans_plusplus_initializer(sample.tolist(),
                                                              self.config["discrete_state"]["k_min"]).initialize()
            else:
                raise NotImplementedError

            # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
            cluster_instance = xmeans(sample.tolist(), initial_centers, kmax=self.config["discrete_state"]["k_max"],
                                      repeat=self.config["discrete_state"]["n_init"])
            cluster_instance.process()

            disc_state = cluster_instance.predict(sample)
            n_clusters = len(cluster_instance.get_clusters())
            self.cluster_centers = cluster_instance.get_centers()
        elif method == "kmeans":
            # sample = self.dados[self.config["states"]].fillna(0)
            n_clusters = self.config["discrete_state"]["n_clusters"]
            cluster_instance = KMeans(n_clusters=n_clusters,
                                      init=self.config["discrete_state"]["init"],
                                      max_iter=self.config["discrete_state"]["max_iter"],
                                      n_init=self.config["discrete_state"]["n_init"],
                                      random_state=0)
            disc_state = cluster_instance.fit_predict(self.dados[self.config["states"]].fillna(0))
            self.cluster_centers = cluster_instance.cluster_centers_
        elif method == "optics":
            # sample = self.dados[self.config["states"]].fillna(0)
            min_samples_frac = len(sample) / self.config["discrete_state"]["k_max"]
            min_samples_frac = 4
            cluster_instance = OPTICS(min_samples=int(min_samples_frac))
            disc_state = cluster_instance.fit_predict(self.dados[self.config["states"]].fillna(0))

            n_clusters = len(np.unique(disc_state))
            self.cluster_centers = None
        else:
            raise NotImplementedError

        logger.info(F"Number of clusters: {n_clusters}")


        self.dados["disc_state"] = disc_state
        self.STATES_columns = ["disc_state"]

        self.prepare_action_probability()

    def prepare_action_probability(self):
        for state in self.dados["disc_state"].unique():
            state_dict = self.dados[self.dados["disc_state"] == state]["action"].value_counts().to_dict()
            self.action_probability_data[state] = defaultdict(lambda: 0, state_dict)

    def action_probability(self, state, action):
        total_acts = np.sum(list(self.action_probability_data[state].values()))
        return self.action_probability_data[state][str(action)] / total_acts

    # def reward(self, seq_id, seq_number):
    #     seq = self.dados.loc[[seq_id]]
    #     max_seq_number = seq.index.get_level_values(1).max()
    #
    #     if seq_number == max_seq_number:
    #         return 1.0
    #     return 0.0

    def save(self):
        file_path = Path(self.config['output_path'], self.config["seed"], "generated_data.json")
        output_list = []

        # Dataset id
        ds = "2019-01-01"

        if ("discrete_state" in self.config) \
                and ("use" in self.config["discrete_state"]) \
                and (self.config["discrete_state"]["use"] is True):
            # Discrete
            for row in self.dados.itertuples():

                # Definir os clusters com os dados de treino e aplicar aos de teste, em seguida converter separadamente
                # para o formato timeline
                idx = getattr(row, "Index")
                seq_id, seq_number = idx

                actions = tuple(self.dados.loc[[row.Index]][self.ACTIONS_columns].to_records(index=False)[0])
                action = self.ACTIONS_INDEXES[actions]

                state = self.dados.loc[[idx]]["disc_state"].values[0]

                state_features = {
                    str(state_idx):
                        getattr(row, feature) for (state_idx, feature) in enumerate(self.STATES_columns)
                }

                reward = self.reward_class.reward(self.dados, seq_id, seq_number)

                row_dict = {
                    'ds': ds,
                    'mdp_id': str(seq_id),
                    'sequence_number': seq_number,
                    "state_features": state_features,
                    "action": str(action),
                    "reward": reward,
                    "action_probability": self.action_probability(state, action),
                    'possible_actions': [str(i) for i in self.ACTIONS_NAMES.keys()],
                    "metrics": {
                        "reward": reward
                    }
                }

                output_list.append(row_dict)

        else:
            raise NotImplementedError

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with file_path.open(mode="w") as file_object:
                for row in output_list:
                    # Save dict data into the JSON file.
                    json.dump(row, file_object)
                    file_object.write("\n")

            logger.info(F"{file_path} created.")

            # Save config
            with Path(self.config['output_path'], self.config["seed"], "students.config.json").open(
                    mode="w") as outfile_config:
                json.dump(self.config, outfile_config, indent=2)

        except FileNotFoundError:
            logger.error(F"{file_path} not found. ")

    def stats(self):
        stats = {
            "Total Students": self.dados.index.get_level_values(self.config["data"]["id"]).nunique(),
            "Total state transitions": self.dados.shape[0]
        }

        logger.info(stats)

    def data_visualization(self, plot="3d"):
        if "output_path" in self.config:
            output = Path(self.config['output_path'], self.config["seed"])
        else:
            output = "/tmp"

        if ("discrete_state" in self.config) and ("use" in self.config["discrete_state"]) \
                and (self.config["discrete_state"]["use"] is True):
            if self.cluster_centers is not None:
                self.discrete_data_visualization(output=output, plot=plot)
            else:
                logger.warning("Clustering without centers. Cannot show visualization.")
        else:
            raise NotImplementedError

    def discrete_data_visualization(self, output, plot):
        if self.cluster_centers is not None:
            self.discrete_data_centers_visualization(output=output, plot=plot)
        else:
            logger.warning("Clustering without centers. Cannot show visualization.")

    def discrete_data_centers_visualization(self, output, plot):
        if plot == "3d":
            self.discrete_data_centers_visualization_3d(output)
        elif plot == "2d":
            self.discrete_data_centers_visualization_2d(output)
        else:
            raise NotImplementedError

    def discrete_data_centers_visualization_2d(self, output):
        # logging.getLogger().isEnabledFor(logging.DEBUG)
        scaler = StandardScaler()
        pca = PCA(n_components=2)

        train_data = self.cluster_centers
        val_data = self.cluster_centers

        scaler.fit(train_data)
        pca.fit(scaler.transform(train_data))
        pca_components = pca.transform(val_data)

        df = pd.DataFrame.from_records(pca_components,
                                       columns=['PCA Component 1', 'PCA Component 2'])
        df["Average dropout"] = pd.concat([self.extras, self.dados["disc_state"]], axis="columns").groupby(
            "disc_state").mean().to_numpy()
        df["Number of records"] = self.dados['disc_state'].fillna(0).value_counts().sort_index().to_numpy()

        # # plot
        fig = plt.figure()
        ax = plt.axes()

        # (c)olor: average dropout in the state
        # (s)ize: number of records in the state
        sc = ax.scatter(df['PCA Component 1'], df['PCA Component 2'],
                        c=df['Average dropout'] * 100,
                        s=df['Number of records'],
                        vmax=100,
                        vmin=0,
                        cmap='winter'
                        )
        plt.colorbar(sc)
        # ax.view_init(30, 185)

        # Save
        df.to_pickle(F"{output}/pca_data_discretization_2d.dataframe.pkl")
        plt.savefig(F"{output}/pca_data_discretization_2d.png")

    def discrete_data_centers_visualization_3d(self, output):

        # logging.getLogger().isEnabledFor(logging.DEBUG)
        scaler = StandardScaler()
        pca = PCA(n_components=3)

        train_data = self.cluster_centers
        val_data = self.cluster_centers

        scaler.fit(train_data)
        pca.fit(scaler.transform(train_data))
        pca_components = pca.transform(val_data)

        df = pd.DataFrame.from_records(pca_components,
                                       columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
        df["Average dropout"] = pd.concat([self.extras, self.dados["disc_state"]], axis="columns").groupby(
            "disc_state").mean().to_numpy()
        df["Number of records"] = self.dados['disc_state'].fillna(0).value_counts().sort_index().to_numpy()

        df.to_pickle(F"{output}/pca_data_discretization_3d.dataframe.pkl")

        # # plot
        plot3d_state_discrete(df, output)


def plot3d_state_discrete(df, output, view=None, normalize_size=True):
    # # plot
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection='3d')
    if view:
        # Example: view={elev:14, azim:-13}
        ax.view_init(**view)

    if normalize_size:
        min_max = (0, 200)
        size = MinMaxScaler(min_max).fit_transform(df[['Number of records']])
    else:
        size = df['Number of records']

    # (c)olor: average dropout in the state
    # (s)ize: number of records in the state
    sc = ax.scatter(df['PCA Component 1'], df['PCA Component 2'], df['PCA Component 3'],
                    c=df['Average dropout'] * 100,
                    s=size,
                    vmax=100,
                    vmin=0,
                    cmap='viridis_r',
                    edgecolors='black',
                    linewidths=0.5,
                    alpha=0.7,
                    )
    plt.colorbar(sc)

    # Save
    plt.savefig(F"{output}/pca_data_discretization_3d.png", bbox_inches="tight")


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in an OpenAI Gym environment."
    )
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")

    parser.add_argument(
        "--seed",
        help="Seed for the test (numpy, torch, and gym).",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--use_gpu",
        help="Use GPU, if available; set the device with CUDA_VISIBLE_DEVICES",
        action="store_true",
    )
    args = parser.parse_args(args)

    if args.seed is None:
        args.seed = 0

    generate_students(args.parameters, seed=args.seed)


def generate_students(config_file, seed=0):
    students = Students(config_file=config_file, seed=seed)

    students.init_data()
    students.save()
    students.stats()
    students.data_visualization(plot="2d")
    students.data_visualization(plot="3d")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    args = sys.argv
    main(args[1:])
