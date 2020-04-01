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
from comet_ml import Experiment
from data_preparation import FederatedModel, MitDataLoader
import os

# %%
# import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.chdir('ecg')

# %%
def run_experiment(clients):
    experiment = Experiment(api_key="H0f8rYHfnX5fAJmQf91EUnB9h",
                        project_name="mit", workspace="friedrich-tim")
    experiment.add_tag('mit_'+str(clients)+'_clients_40_epochs_locale_1')

    model = FederatedModel(MitDataLoader, "report/model/mit_"+str(clients)+"_clients_40_epochs_locale_1", "examples/cinc17/config.json", clients, experiment)
    model.fit(40, 1)


# %%
def run_experiments(client_counts):
    for client_count in client_counts:
        run_experiment(client_count)
        #try:
        #    run_experiment(client_count)
        #except Exception as e:
        #    print(e)
        #    print('failed to run ', client_count)


# %%
# experiments = [2, 3, 4, 5]
experiments = [1]
run_experiments(experiments)

# %%
