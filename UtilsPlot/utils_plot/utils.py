import pandas as pd
# !pip install tueplots
# !pip install latex
# !sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
import latex
from tueplots import bundles
pd.set_option("display.max_rows", None, "display.max_columns", None)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update(bundles.neurips2023( usetex=True))
default_fig_size = plt.rcParams.get('figure.figsize')
import os
import tqdm
import glob


def filter_list(list_dir, filter_value):
    list_dir_filtered = []
    for dir in list_dir :
        if filter_value in dir :
            list_dir_filtered.append(dir)
    return list_dir_filtered


def get_list_dir(results_dir,):
    list_dir = glob.glob(os.path.join(results_dir,"**/hparams.yaml"), recursive=True)
    return list_dir


liste_of_params = ['base_dist', "training_type", "network", "optimizer", "proposal", "seed", "lightning_logs", "version", ]
def create_dictionnary_params(list_dir, results_dir):
    nb_parameter = list_dir[0].strip(results_dir).count("/")
    dictionnary_params = {key: [] for key in liste_of_params[:nb_parameter]}
    dictionnary_params['folder_event'] = []
    for path in list_dir :
        if 'seed' not in path :
            continue
        current_nb_parameter = path.strip(results_dir).count("/")
        list_param = path.strip(results_dir).split("/")
        assert not (current_nb_parameter>len(liste_of_params)), f"The number of parameter {current_nb_parameter} is not the same as the number of parameter in the list {len(liste_of_params)}"
        if current_nb_parameter < nb_parameter :
            list_param = ['None']*(nb_parameter-current_nb_parameter) + list_param
        for key, param in zip(liste_of_params[:nb_parameter], list_param) :
            dictionnary_params[key].append(param)
        dictionnary_params['folder_event'].append(path.strip("hparams.yaml"))
    return dictionnary_params
    

