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
from data_preparation import FederatedModel, MitDataLoader, CombinedDataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
experiment = Experiment(api_key="H0f8rYHfnX5fAJmQf91EUnB9h",
                        project_name="mit", workspace="friedrich-tim")
experiment.add_tag('baseline_mit_no_averaging')

model = FederatedModel(MitDataLoader, "report/model/test_combined", "examples/cinc17/config.json", 1)
model.fit(1, 30)

# %%
experiment = Experiment(api_key="H0f8rYHfnX5fAJmQf91EUnB9h",
                        project_name="cinc", workspace="friedrich-tim")
experiment.add_tag('baseline_mit_baseline_40_epochs_1_locale')

model = FederatedModel(MitDataLoader, "report/model/test_combined", "examples/cinc17/config.json", 1)
model.fit(30, 1)

# %%
experiment = Experiment(api_key="H0f8rYHfnX5fAJmQf91EUnB9h",
                        project_name="cinc", workspace="friedrich-tim")
experiment.add_tag('mit_2_clients_40_epochs_locale_1')

model = FederatedModel(MitDataLoader, "report/model/test_combined", "examples/cinc17/config.json", 5)
model.fit(1, 1)

# %%
max(15/2, 5)

# %%
len(model.data_loader.get_data_for_client(1)['train_x'])

# %%
len(model.data_loader.get_data_for_client(2)['train_x'])

# %%
len(model.data_loader.get_data_for_client(3)['train_x'])

# %%
len(model.data_loader.get_data_for_client(4)['train_x'])

# %%
len(model.data_loader.get_data_for_client(0)['train_x'])

# %%
experiment = Experiment(api_key="H0f8rYHfnX5fAJmQf91EUnB9h",
                        project_name="mit", workspace="friedrich-tim")
experiment.add_tag('combined_test')

model = FederatedModel(CombinedDataLoader, "report/model/test_combined", "examples/cinc17/config.json", 10, experiment)
model.fit(40, 1)

# %%
