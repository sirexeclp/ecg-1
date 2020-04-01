# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from data_preparation import FederatedModel, CincDataLoader, CombinedDataLoader, MitDataLoader
import os
import tensorflow.keras as keras
import tensorflow
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import tqdm
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.chdir('ecg')

# %%
def read_config( config_path):
    with open(config_path,"r") as f:
        params = json.load(f)
    params.update({
    "input_shape": [None, 1],
    "num_categories": 4
    })
    return params

cinc_data_loader = CincDataLoader(1, read_config("examples/cinc17/config.json"))

# %%
mit_data_loader = MitDataLoader(2, None)

# %%
combined_data_loader = CombinedDataLoader(4, read_config("examples/cinc17/config.json"))


# %%
def f1_score(conf_mat):
    true_positives = np.diagonal(conf_mat)#[rhythm_type,rhythm_type]
    sum_reference = conf_mat.sum(axis=1)#[rhythm_type]
    sum_predicted = conf_mat.sum(axis=0)#[rhythm_type]
    return 2 * true_positives / (sum_reference + sum_predicted)

def onehot2index(y):
    return np.argmax(y, axis=2).reshape(-1)

def evaluate(model, val_x, val_y):
    val_pred = model.predict(val_x)
    val_pred_oh = onehot2index(val_pred)

    val_true = val_y
    val_true_oh = onehot2index(val_true)

    conf_mat = confusion_matrix(y_true=val_true_oh, y_pred=val_pred_oh)
    scores = f1_score(conf_mat=conf_mat)
    
    losses = [tensorflow.keras.backend.get_value(lf(val_true, val_pred)) for lf in model.loss_functions]
    
    true_positives = np.diagonal(conf_mat)
    
    acc = np.sum(true_positives) / np.sum(conf_mat)
    
    result = {
        "losses":losses
        ,"scores": scores
        ,"conf_mat": conf_mat
        ,"acc": acc
    }
    return result


# %%
model_path = f"report/model/combined_4_clients_40_epochs_locale_1/best_model.hdf5" #model architecture is the same for all
model = keras.models.load_model(model_path)


# %%
def validate(clients, experiment, test_set):
    model_path = f"report/model/{experiment}_{clients}_clients_40_epochs_locale_1/best_model.hdf5"
    #print("loading weights")
    model.load_weights(model_path)
    #print(test_set)
    loader = data_loaders[test_set]
    #print(loader)
    x_test = np.array(loader.get_data_for_client(0)["test_x"]) # test data is the same for all clients
    y_test = np.array(loader.get_data_for_client(0)["test_y"])
    #print(f"evaluating {clients}")
    result=evaluate(model, x_test, y_test)
    return result
    #model.fit(40, 1)


# %%
def run_validation(experiments, dataset, test_set):
    results = []
    for client_count in tqdm.tqdm(experiments):
        tmp = validate(client_count, dataset, test_set)
        tmp["dataset"] = dataset
        tmp["client_count"] = client_count
        tmp["test_set"] = test_set
        results.append(tmp)
    return results


# %%
all_results = []
data_loaders = {"mit":mit_data_loader
               ,"cinc":cinc_data_loader
               ,"combined":combined_data_loader}

# %%
#mit_data_loader.get_data_for_client(0)["test_y"])

# %%
for key, value in data_loaders.items():
    experiments = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]#, 17, 18, 19, 20]
    all_results += run_validation(experiments, dataset="cinc", test_set=key)
    
    experiments = [1,2, 3, 4, 5]
    all_results += run_validation(experiments, dataset="mit", test_set=key)
    
    experiments = [1,4, 6, 8, 10]
    all_results += run_validation(experiments, dataset="combined", test_set=key)

# %%
results = pd.DataFrame(all_results)

# %%
results["mean_f1"] = results.scores.apply(np.mean)

# %%
results = results.rename(columns={"acc":"Test Accuracy"
                                 ,"mean_f1": "Test F1-Score"})
results.tail()

# %%
results[results["client_count"]==1]

# %%
results.to_csv("report/test_results_2.csv")


# %%
def plot_metric(metric):
    for index, dataset in results.groupby("dataset"):
        title = f"[{index}] {metric} vs. #Clients"
        for jdex, test_set in dataset.groupby("test_set"):
            plt.scatter(test_set["client_count"], test_set[metric], label=f"Tested on: {jdex}")
            plt.title(title)
            plt.xticks(dataset["client_count"])
            plt.ylabel(metric)
            plt.xlabel("#Clients")
            plt.ylim(0,1)
            plt.legend()
            #plt.show()
            
        
        fig_path = Path("report/img/final")
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path/Path(f"{index}-{metric}").with_suffix(".pdf"),bbox_inches='tight')
        plt.show()

# %%
style = """
font.size: 16
axes.titlesize : 24
axes.labelsize : 20
lines.linewidth : 3
lines.markersize : 10
xtick.labelsize : 16
ytick.labelsize : 16
legend.fontsize: 16
figure.figsize: 16, 9
"""
style_path = Path("matplot_stype.mplstyle")
style_path.write_text(style)

# %%
with plt.style.context(["seaborn-colorblind",str(style_path)]):
    plot_metric("Test F1-Score")
