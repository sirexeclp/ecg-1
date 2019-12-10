# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
os.chdir("ecg")
import keras as keras
import tensorflow as tf
import json
import load
import network
import tensorflow_federated as tff
from tensorflow.keras.optimizers import Adam
import sys,os
import numpy as np
import collections
# %load_ext autoreload
# %autoreload 2
os.chdir('../')

# %%
data_json, model_path = "examples/cinc17/dev.json","../saved/cinc17/1575022999-481/0.384-0.890-019-0.301-0.898.hdf5"
#model = keras.models.load_model(model_path)
#

# %%
params = json.load(open("examples/cinc17/config.json", 'r'))
print("Loading training set...")
train = load.load_dataset(params['train'])
print("Loading dev set...")
dev = load.load_dataset(params['dev'])
print("Building preprocessor...")
preproc = load.Preproc(*train)
print("Training size: " + str(len(train[0])) + " examples.")
print("Dev size: " + str(len(dev[0])) + " examples.")
params.update({
    "input_shape": [None, 1],
    "num_categories": len(preproc.classes)
})

model = network.build_network(**params)

# %%
train_x, train_y = preproc.process(*train)
dev_x, dev_y = preproc.process(*dev)

# %%
NUM_CLIENTS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

def preprocess(dataset):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', train_x),
        ('y', train_y),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)


# %%
class FederatedDataLoader:
    
    def __init__(self, data_json, clients=10):
        self.allData = load.load_dataset(data_json)
        self.clients = clients
        self.groupClientData()
    
    def clientData(self, index):
        return self.clientsData[index]
    
    def groupClientData(self):
        self.clientsData = []
        xSplit = self.splitArray(self.allData[0])
        ySplit = self.splitArray(self.allData[1])
        
        for index in range(self.clients):
            self.clientsData.append([xSplit[index], ySplit[index]])

    def splitArray(self, array):
        splitLen = len(array)/self.clients
        return np.array_split(array, splitLen)
loader = FederatedDataLoader("examples/cinc17/dev.json")

# %%
loader.clientData(0)


# %%
def preprocess(dataset):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', element[0]),
        ('y', element[1]),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)


# %%
preprocessed_example_dataset = preprocess(tf.data.Dataset.from_generator(lambda: loader.clientsData,tf.float64))

# %%
sample_batch = tf.nest.map_structure(
    lambda x: x, next(iter(loader.clientsData)))

# %%
from tensorflow.compat.v2 import convert_to_tensor
# Wrap a Keras model for use with TFF.
def model_fn():
    model = network.build_network(**params)
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=[tf.keras.metrics.Accuracy()])
    return tff.learning.from_compiled_keras_model(model, convert_to_tensor(sample_batch))


# %%
model_fn()

# %% [markdown]
# iterative_process = tff.learning.build_federated_averaging_process(model_fn)
# state = iterative_process.initialize()
# for round_num in range(2, 11):
#   state, metrics = iterative_process.next(state, loader.clientsData)
#   print('round {:2d}, metrics={}'.format(round_num, metrics))

# %%
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
import tensorflow_federated as tff

# Load simulation data.
source, _ = tff.simulation.datasets.emnist.load_data()
def client_data(n):
  return source.create_tf_dataset_for_client(source.client_ids[n]).map(
      lambda e: {
          'x': tf.reshape(e['pixels'], [-1]),
          'y': e['label'],
  }).repeat(10).batch(20)

# Pick a subset of client devices to participate in training.


# %%
train_data = [client_data(n) for n in range(3)]

# Grab a single batch of data so that TFF knows what data looks like.
sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(train_data[0]).next())

# %%
from keras import backend
len(backend.tensorflow_backend._get_available_gpus())

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
