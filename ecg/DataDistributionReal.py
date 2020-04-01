# -*- coding: utf-8 -*-
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
# %load_ext autoreload
# %autoreload 2
import os
#os.chdir("ecg")
from data_preparation import MitDataLoader, CombinedDataLoader, CincDataLoader
from sklearn.model_selection import train_test_split
import json
import logging
import load
import numpy as np
import matplotlib.pyplot as plt
from MITDBDataProvider import *
os.chdir('../')


# %%
def onehot2index(y):
    return np.argmax(y, axis=2).reshape(-1)

def class2index(y):
    y_int = [[pipeline.preproc.class_to_int[c] for c in sli] for sli in y]
    y_int = [x for sli in y_int for x in sli]
    return np.array(y_int).reshape(-1)


# %%
#pipeline = CincPipeline()
os.chdir('ecg')
print(os.getcwd())
#result,train, test = pipeline.execute("examples/cinc17/config.json")


# %%
import matplotlib
matplotlib.rcParams["figure.figsize"] = 16,9
int_to_class = ["A","N","O","~"]
def autolabel(rects):
    # Attach some text labels.
    for rect in rects:
        if rect.get_height() > 0:
            plt.text(rect.get_x() + rect.get_width() / 2.,
                    rect.get_y() + rect.get_height() / 2. - 20,
                    '%d'%rect.get_height(),
                    ha = 'center',
                    va = 'center')

def plot_data_hist(y, description):
    
    #y_int = []
    #for i in y:
    #    y_int.append(pipeline.preproc.class_to_int[i[0]])
    #[pipeline.preproc.class_to_int[j] for j in y for x in y]onehot2index(y)
    bins = np.bincount(y)
    lbls = [int_to_class[x] for x,_ in enumerate(bins)]
    
    plt.title(f"Distribution of Training Classes ({description})")
    plt.xlabel("Class")
    plt.ylabel("# of Examples")
    plt.bar(lbls, bins,width=0.75, align='center')
    print(np.sum(bins,axis=0))
    #plt.show()

def plot_data_pie(y, description):
    bins = np.bincount(y)
    lbls = [pipeline.preproc.int_to_class[x] for x,_ in enumerate(bins)]
    plt.title(f"Distribution of Training Classes ({description})")
    explode = (0.1, 0.05, 0.05, 0.05)
    plt.pie(bins, labels= lbls,autopct='%1.1f%%',
            shadow=True,startangle=90, explode=explode,)
    plt.show()

def plot_data_non_pie(y, description, data_type, absolute=True, labels=None):
    if labels is None:
        plt.xlabel("Client")
        labels = np.arange(len(y))
    
    num_classes = 4
    bins = np.zeros((len(y),num_classes))
    for index, item in enumerate(y):
        tmp = np.bincount(item)
        bins[index,:len(tmp)] = tmp
    #bins = np.array(list(zip(*[np.bincount(i) for i in y])))
    lbls = [int_to_class[x] for x in range(num_classes)]
    #print([np.bincount(i) for i in y])
    plt.title(f"[{description}] Distribution of {data_type} Classes per Client")
    
    plt.ylabel("# of Examples")
    plt.xticks(*zip(*enumerate(labels)))

    b_last = np.zeros(len(y))
    for index, b in enumerate(bins.T):
        b = np.array(b)
        if not absolute:
            b = b/np.sum(bins,axis=0) 
        #print(labels,b,b_last)
        r = plt.bar(labels, b,width=0.75, align='center', bottom=b_last)
        autolabel(r)
        b_last += b
        #print(np.sum(bins,axis=1))
    plt.legend(lbls)


# %%
def read_config( config_path):
    with open(config_path,"r") as f:
        params = json.load(f)
    params.update({
    "input_shape": [None, 1],
    "num_categories": 4
    })
    return params


# %%
combined_loader = CombinedDataLoader(10, read_config("examples/cinc17/config.json"))

# %%
mit_loader = MitDataLoader(5,{})

# %%
cinc_loader = CincDataLoader(16, read_config("examples/cinc17/config.json"))


# %%
def plot_dist(data, data_set, data_type, x_tics):
    style_path = Path("matplot_stype.mplstyle")
    with plt.style.context(["seaborn-colorblind",str(style_path)]):
        plot_data_non_pie(data
                      , data_set, data_type,True, x_tics)
        fig_path = Path("report/img/final")
        fig_path.mkdir(exist_ok=True)
        plt.savefig(fig_path/Path(f"{data_set}-{data_type.lower()}").with_suffix(".pdf"),bbox_inches='tight')
        plt.show()
        
def plot_train_dist(loader, data_set):
    data_type = "Training"
    data = [onehot2index(loader.get_data_for_client(d)["train_y"]) for d in range(loader.client_count)]
    plot_dist(data, data_set, data_type,None)
    
def plot_val_test_dist(loader, data_set):
    data_type = "Validation and Test"
    data=[onehot2index(loader.get_data_for_client(0)["val_y"]),
        onehot2index(loader.get_data_for_client(0)["test_y"])]
    plot_dist(data, data_set, data_type, ["Validation", "Test"])


# %%
plt.plot(mit_loader.get_data_for_client(0)["train_x"][0])
np.std(mit_loader.get_data_for_client(0)["train_x"][3])

# %%
cinc_loader.get_data_for_client(4)["train_x"].shape
#def z_transform(inp):
#    return np.array([(i - np.mean(i))/np.std(i) for i in inp])

# %%
mit_data_raw = get_data_4_fed("/workspace/telemed5000/code/data/",
               record_list=["201", "202","203", "219", "222"])

# %%
train_x, test_x, train_y, test_y = mit_data_raw

# %%
len(test_y)

# %% [markdown]
# np.savez("test_mit_data",test_x=mit_loader.get_data_for_client(4)["test_x"],
#        test_y=mit_loader.get_data_for_client(4)["test_y"])

# %% [markdown]
# np.savez("test_cinc_data",test_x=cinc_loader.get_data_for_client(4)["test_x"],
#         test_y=cinc_loader.get_data_for_client(4)["test_y"])

# %% [markdown]
# np.savez("test_combined_data",test_x=combined_loader.get_data_for_client(4)["test_x"],
#          test_y=combined_loader.get_data_for_client(4)["test_y"])

# %%
loaded = np.load("test_mit_data.npz")
assert np.array_equal(loaded["test_x"], mit_loader.get_data_for_client(4)["test_x"])
assert np.array_equal(loaded["test_y"], mit_loader.get_data_for_client(4)["test_y"])
print("OK")

# %%
loaded = np.load("test_cinc_data.npz")
assert np.array_equal(loaded["test_x"], cinc_loader.get_data_for_client(4)["test_x"])
assert np.array_equal(loaded["test_y"], cinc_loader.get_data_for_client(4)["test_y"])
print("OK")

# %%
loaded = np.load("test_combined_data.npz")
assert np.array_equal(loaded["test_x"], combined_loader.get_data_for_client(4)["test_x"])
assert np.array_equal(loaded["test_y"], combined_loader.get_data_for_client(4)["test_y"])
print("OK")

# %%
np.random.randint(len(loaded["test_x"]), size=10)

# %%
#z = z_transform(mit_loader.get_data_for_client(0)["train_x"])
#z[0].std()

# %%
cinc_loader.get_data_for_client(0)["train_x"][7].std()

# %%
plot_train_dist(cinc_loader, "cinc")
plot_val_test_dist(cinc_loader, "cinc")

# %%
plot_train_dist(mit_loader, "mit")
plot_val_test_dist(mit_loader, "mit")

# %%
plot_train_dist(combined_loader, "combined")
plot_val_test_dist(combined_loader, "combined")

# %%
plot_train_dist(combined_loader, "combined")
plot_val_test_dist(combined_loader, "combined")

# %%
x

# %%
#
data=[onehot2index(combined_data_loader.get_data_for_client(0)["test_y"]), onehot2index(combined_data_loader.get_data_for_client(0)["val_y"])]
style_path = Path("matplot_stype.mplstyle")
with plt.style.context(["seaborn-colorblind",str(style_path)]):
    plot_data_non_pie(data
                  ,range(len(data))
                  , "", True)

# %%

# %%
for i in range(5):
    data = data_loader.get_data_for_client(i)
    train_y = data["train_y"]
    plot_data_hist(onehot2index(train_y), "train_after")
    plt.show()

# %%
train_y.shape
data["test_y"]

# %%
np.array(y_train[0]).shape

# %%
plot_data_hist(onehot2index(y_test), "train_after")

# %%
for cli in y_train:
    plot_data_hist(onehot2index(cli), "train_after")
    plt.show()

# %%
y_train_0 = y_train[0]

# %%
x_train_0 = x_train[0]

# %%
classes = np.argmax(np.array(y_train_0),axis=2)

# %%
classes[:,0]

# %%
classes=np.array([np.bincount(c).argmax() for c in classes])

# %%
class_counts=np.bincount(classes)
class_counts
max_class = np.max(class_counts)
oversample = max_class - class_counts
oversample
idx = [np.random.choice(*np.where(classes == c),over)
       for c,over in enumerate(oversample)
                       if over > 0 ]
idx

# %%
for c,over in enumerate(oversample):
    if over >0:
        #print(over)
        #print(len(np.where(classes==c)))
        print(np.random.choice(*np.where(classes == c),over))


# %%
def do_oversampling(x, y):
    x = np.array(x)
    y = np.array(y)
    
    classes = np.argmax(y,axis=2)
    classes = np.array([np.bincount(c).argmax() for c in classes])
    class_counts=np.bincount(classes)
    max_class = np.max(class_counts)
    oversample = np.array(max_class - class_counts, dtype=np.int32)

    result_y = y.copy()
    result_x = x.copy()
    idx = [np.random.choice(*np.where(classes == c),over)
           for c,over in enumerate(oversample)
                           if over > 0 ]
    for i in idx:
        print(i)
        print(i.dtype)
        result_y = np.concatenate([result_y, y[i]])
        result_x = np.concatenate([result_x, x[i]])
    return result_x, result_y


# %%
a,b = do_oversampling(x_train_0, y_train_0)

# %%
classes = np.argmax(b,axis=2)
#classes = np.array([np.bincount(c).argmax() for c in classes])
class_counts=np.bincount(classes.reshape(-1))
class_counts

# %%
plot_data_non_pie([class2index(train[1]),onehot2index(result["train_y"])]
                  ,["train_before", "train_after"]
                  , "train_before", False)

# %%
matplotlib.rcParams.update({'font.size': 16})
plot_data_hist(class2index(train[1]), "train_before")
plot_data_hist(onehot2index(result["train_y"]), "train_after")
plt.legend(["before", "after"])

# %%
plot_data_non_pie(class2index(train[1]), "train_before")
plot_data_pie(onehot2index(result["train_y"]), "train_after")

bins1 = np.bincount(class2index(train[1]))
bins2 = np.bincount(onehot2index(result["train_y"]))
bins = bins1-bins2
lbls = [pipeline.preproc.int_to_class[x] for x,_ in enumerate(bins)]
plt.title("Distribution of Training Classes ()")
explode = (0.1, 0.05, 0.05, 0.05)
plt.pie(bins, labels= lbls,autopct='%1.1f%%',
        shadow=True,startangle=90, explode=explode,)
plt.show()

# %%
plot_data_dist(onehot2index(result["train_y"]), "train_after")

# %%
plt.plot(result["test_x"][0])
pipeline.preproc.int_to_class[result["test_y"][0][0]]

# %%
result["train_y"][0][0]

# %%
plot_data_dist(result["train_y"], "train")

# %%
fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, explode=explode,)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# %%
import tensorflow
tensorflow.keras.utils.to_categorical(
                ["B","A","C"] , num_classes=3)

# %%
#os.chdir("../")
from MITDBDataProvider import *


# %%
#data = get_data_4_fed("/workspace/telemed5000/code/data/", record_list=["202"])

# %%
def f1_score(conf_mat):
    true_positives = np.diagonal(conf_mat)#[rhythm_type,rhythm_type]
    sum_reference = conf_mat.sum(axis=1)#[rhythm_type]
    sum_predicted = conf_mat.sum(axis=0)#[rhythm_type]
    return 2 * true_positives / (sum_reference + sum_predicted)

def evaluate(model, val_x, val_y):
    val_pred = model.predict(val_x)
    val_pred = onehot2index(val_pred)

    val_true = val_y
    val_true = model.onehot2index(val_true)

    conf_mat = confusion_matrix(y_true=val_true, y_pred=val_pred)
    scores = f1_score(conf_mat=conf_mat)
    
    return scores, conf_mat
    #history = self.model.history.history
    #history.setdefault("val_f1", []).append(scores)
    #history.setdefault("val_conf_mat", []).append(conf_mat)
    #print("— val_f1: {:.3f}".format(scores.mean()))
