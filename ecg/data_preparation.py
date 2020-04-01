# -*- coding: utf-8 -*-
# +
import logging
# %load_ext autoreload
# %autoreload 2

# import comet_ml in the top of your file
from comet_ml import Experiment
    
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="H0f8rYHfnX5fAJmQf91EUnB9h",
                        project_name="telemed", workspace="friedrich-tim")



# +
import tensorflow.keras
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,Callback
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import random


os.chdir("/workspace/telemed5000/ecg/ecg")
import network
import load
from MITDBDataProvider import *
os.chdir('../')
# -

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# +
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


# -

# # Data Loader

from sklearn.model_selection import train_test_split
class CincDataLoader():
    
    def __init__(self, num_clients, params):
        print('clients ', num_clients)
        self.num_clients = num_clients
        self.client_count = self.num_clients #compatibility with other data_loaders
        self.data = self.execute(params)
        
        
    @staticmethod
    def load_train_set(params):
        logging.info(f"Loading training set...{params['train']}")
        train = load.load_dataset(params['train'])
        return train
    
    @staticmethod
    def load_test_set(params):
        logging.info(f"Loading dev set... {params['dev']}")
        dev = load.load_dataset(params['dev'])
        return dev
    
    @staticmethod
    def build_preprocessor(train):
        logging.info("Building preprocessor...")
        preproc = load.Preproc(*train)
        return preproc
    
    def get_data_for_client(self, client_id):
        return self.data[client_id]

    def execute(self, params):
        result = []
        train = self.load_train_set(params)
        test = self.load_test_set(params)
        
        preproc = self.build_preprocessor(train)
        train_x, train_y = preproc.process(*train)
        test_x, test_y = preproc.process(*test)
        
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=0.2, random_state=42)
        
        for client_id in range(self.num_clients):
            result.append({
                "train_x": np.array_split(train_x, self.num_clients)[client_id]
               ,"train_y": np.array_split(train_y, self.num_clients)[client_id]
               ,"val_x": val_x
               ,"val_y": val_y
               ,"test_x": test_x
               ,"test_y": test_y
               ,"params": params})
        return result


class MitDataLoader():
    
    def __init__(self, client_count, params):
        # x train x test y train y test
        self.file_path = "/workspace/telemed5000/code/data/"
        self.data = self.execute()
        self.params = params
        self.client_count = client_count

    def execute(self):
        reccords_with_afib_and_sinus = ["201", "202",
                                        "203", "219", "222"]
        
        train_x, test_x, train_y, test_y = get_data_4_fed(self.file_path
                                   ,record_list=reccords_with_afib_and_sinus)
        
        #def z_transform(inp):
        #    return (inp - np.mean(inp, axis=1))/np.std(inp, axis=1)
        
        #train_x = [z_transform(i) for i in train_x]
        

        test_x, val_x, test_y, val_y = train_test_split(
            test_x, test_y, test_size=0.5, random_state=42)
        
        
        train_x, train_y = zip(*[self.do_oversampling(np.array(x),np.array(y))
                            for x,y in zip(train_x, train_y)])

        self.train_x = self.prepare_x_train(train_x)
        self.train_y = train_y
        
        self.val_x = np.expand_dims(self.prepare_x(val_x), axis=2)
        self.val_y = np.array(val_y)
        
        self.test_x = np.expand_dims(test_x, axis=2)
        self.test_y = test_y

    @staticmethod
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
                               if class_counts[c] > 0 ]
        for i in idx:
            #print(i)
            #print(i.dtype)
            result_y = np.concatenate([result_y, y[i]])
            result_x = np.concatenate([result_x, x[i]])
        return result_x, result_y
    
    def prepare_x(self, data_chunk):
        return data_chunk
    
    def prepare_x_train(self, data):
        return [self.prepare_x(x_1) for x_1 in data]        
        
    def flatten_array(self, array):
        array = np.array(array)
        return array.reshape(-1, array.shape[-1])
    
    def save_test(self):
        path = "report/test_mit"
        np.savez(path,
                 test_x=self.test_x,
                 test_y=self.test_y)

    
    def get_data_for_client(self, client_id):
        if self.client_count <= 1:
            #num_clients = np.array(self.train_y).shape[0]
            #min_len = min([len(client) for client in self.train_x])
            #x = [self.train_x[i][:min_len] for i in range(5) ]
            x = flatten_list(self.train_x) # self.flatten_array(x)
            
            # data = [np.expand_dims(np.array(self.train_x[client_id]),axis=2), np.array(self.train_y[client_id])]
             
            y = flatten_list(self.train_y) #np.array([np.array(self.train_y)[i][:min_len] for i in range(5)]).reshape(min_len * num_clients, 35, 4)
            
            # y = np.array(self.train_y).reshape(195, 35, 4)
            x = np.expand_dims(x, axis=2)
        else:
            data = [np.expand_dims(np.array(self.train_x[client_id]),axis=2), np.array(self.train_y[client_id])]
            x = data[0]
            y = data[1]
            combined = list(zip(x,y))
            random.shuffle(combined)
            min_len = min([len(client) for client in self.train_x])
            x, y = zip(*combined)
            x = x[:min_len]
            y = y[:min_len]
            
        return {
                "train_x": np.array(x)
               ,"train_y": np.array(y)
               ,"val_x": self.val_x
               ,"val_y": self.val_y
               ,"test_x": self.test_x
               ,"test_y": self.test_y
               ,"params": self.params }


# +
class CombinedDataLoader():
    def __init__(self, client_count, params):
        self.mit_data_loader = MitDataLoader(max(client_count//2, 1), params)
        self.cinc_data_loader = CincDataLoader(max(client_count//2, 1), params)
        self.min_len = self.calculate_max_size()
        self.client_count = client_count
        
    def calculate_max_size(self):
        min_len_mit = min([len(client) for client in self.mit_data_loader.train_x])
        return min_len_mit
    
    def get_data_for_client(self, client_id):
        
        mit_data = self.mit_data_loader.get_data_for_client(0)
        cinc_data = self.cinc_data_loader.get_data_for_client(0)
        
        def combine_x_y(data, data_type):
            np.random.seed(42)
            xs = [i[f"{data_type}_x"] for i in data]
            ys = [i[f"{data_type}_y"] for i in data]
            lens = [len(i) for i in xs]
            min_len = min(lens)
            #miny = min([len(i) for i in ys])
            indices = [np.random.randint(i, size=min_len) for i in lens]
            xs = [i[idx] for idx, i in zip(indices,xs)]
            ys = [i[idx] for idx, i in zip(indices,ys)]
            return np.concatenate(xs), np.concatenate(ys)
        
        val_x, val_y = combine_x_y([mit_data, cinc_data], "val")
        test_x, test_y = combine_x_y([mit_data, cinc_data], "test")
        
#         mit_x = mit_data['val_x']
#         mit_y = mit_data['val_y']
#         cinc_x = cinc_data['val_x']
#         cinc_y = cinc_data['val_y']

#         #combined = list(zip(cinc_x, cinc_y))
#         #random.shuffle(combined)
#         #cinc_x, cinc_y = zip(*combined)
#         val_x = np.concatenate([mit_x, cinc_x])#[: len(mit_x)]])
#         val_y = np.concatenate([mit_y, cinc_y])#[: len(mit_y)]])
        
        if self.client_count == 1:
            
            mit_x = mit_data['train_x']
            mit_y = mit_data['train_y']
            cinc_x = cinc_data['train_x']
            cinc_y = cinc_data['train_y']

            combined = list(zip(cinc_x, cinc_y))
            random.shuffle(combined)
            cinc_x, cinc_y = zip(*combined)
            x = np.concatenate([mit_x, cinc_x[: len(mit_x)]])
            y = np.concatenate([mit_y, cinc_y[: len(mit_y)]])
            
            
           
            
            data = {
                'train_x': x,
                'train_y': y,
                'val_x': val_x,
                'val_y': val_y,
                "test_x": test_x,
                "test_y": test_y
            }
            return data
        
        if client_id%2 == 0:
            loader = self.mit_data_loader
        else:
            loader = self.cinc_data_loader
        
        data = loader.get_data_for_client(client_id//2)
        
        x = data['train_x']
        y = data['train_y']
        combined = list(zip(x,y))
        random.shuffle(combined)
        x, y = zip(*combined)
        
        data['train_x'] = np.array(x[:self.min_len])
        data['train_y'] = np.array(y[:self.min_len])
        data["val_x"] = val_x
        data["val_y"] = val_y
        data["test_x"] = test_x
        data["test_y"] = test_y
        return data


# -

# # Federated Model

class FederatedModel():
    def __init__(self, data_loader_class, result_path, config_path, num_clients, experiment):
        self.experiment = experiment
        self.result_path = Path(result_path)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.params = self.read_config(config_path)
        self.data_loader = data_loader_class(num_clients, self.params)
        self.num_clients = num_clients

        self.callbacks = {}
        
        self.models = self.build_models()
        self.best_loss = np.inf
    
    def build_models(self):
        models = []
        for client_id in range(self.num_clients):
            models.append(self.build_model())
        return models
            
    def read_config(self, config_path):
        with open(config_path,"r") as f:
            params = json.load(f)
        params.update({
        "input_shape": [None, 1],
        "num_categories": 4
        })
        return params
    
    @staticmethod
    def onehot2index(y):
        return np.argmax(y, axis=2).reshape(-1)
    
    def build_model(self):
        self.batch_size = self.params.get("batch_size", 32)
        
        model = network.build_network(**self.params)
        self.optimizer = Adam(
            lr=self.params["learning_rate"],
            clipnorm=self.params.get("clipnorm", 1))

        model.compile(loss='categorical_crossentropy',
                          optimizer=self.optimizer,
                          metrics=['accuracy'])
        
        model.onehot2index = self.onehot2index
        
        #self.build_callbacks()
        return model
    
    def build_callbacks(self):
        result_path = self.result_path / "best_model.hdf5"
        checkpointer = ModelCheckpoint(str(result_path)
                , monitor='val_loss', verbose=1
                ,save_best_only=True, mode='min', period=1)
        
        stopping = tensorflow.keras.callbacks.EarlyStopping(patience=8)

        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
            factor=0.1,
            patience=2,
            min_lr=self.params["learning_rate"] * 0.001)
        
        f1_score = F1Score(self.params["num_categories"])
        
        self.callbacks = {
            "checkpointer": checkpointer
            ,"stopping": stopping
            ,"reduce_lr": reduce_lr
            ,"f1_score": f1_score
        }
    
    def save_history(self):
        history_path = self.result_path / "history.csv"
        df = pd.DataFrame(self.history)
        df.to_csv(history_path)
    
    def save_model(self, model):
        result_path = self.result_path / "best_model.hdf5"
        model.save(result_path)
    
    def calculate_averaged_weights(self):
        weights = [model.get_weights() for model in self.models]
        new_weights = list()

        for weights_list_tuple in zip(*weights):
            new_weights.append(
                np.array([np.array(weights_).mean(axis=0)\
                    for weights_ in zip(*weights_list_tuple)]))
        return new_weights
    
    def update_all_weights(self, weights):
        for model in self.models:
            model.set_weights(weights)
            
    def add_metrics(self, metrics, model, context):
        # Cinc
        cinc_val_x_data = self.data_loader.get_data_for_client(0)["val_x"]
        cinc_val_y_data = self.data_loader.get_data_for_client(0)["val_y"]
        #score = model.evaluate(cinc_val_x_data, cinc_val_y_data, verbose=0)
        
        scores = evaluate(model, cinc_val_x_data, cinc_val_y_data)
        metrics[str(context) + "_val_acc"] = scores['acc']
        metrics[str(context) + '_val_loss'] = scores['losses'][0]
        metrics[str(context) + '_matrix'] = scores['conf_mat']
        metrics[str(context) + '_scores'] = scores['scores']
        return metrics
    
    def fit(self, max_epochs=100, locale_epochs=1):
        
        #self.data_loader.save_test()
        
        self.history = []
        for epoch in range(max_epochs):
            metrics = { }
            for client_id in range(self.num_clients):
                data = self.data_loader.get_data_for_client(client_id)
                model = self.models[client_id]
                model.val_x = data["val_x"]
                model.val_y = data["val_y"]
                history = model.fit(
                    data["train_x"]
                    ,data["train_y"]
                    ,batch_size=self.batch_size
                    ,epochs=locale_epochs)
                    #,callbacks=list(self.callbacks.values())
                    #,validation_data=(data["val_x"], data["val_y"]))
                
                metrics = self.add_metrics(metrics, model, client_id)
                
            weights = self.calculate_averaged_weights()
            self.update_all_weights(weights)
            
            metrics = self.add_metrics(metrics, self.models[0], 'combined')
            self.experiment.log_metrics(metrics, epoch=epoch, step=epoch)
            metrics['epoch'] = epoch
            self.history.append(metrics)
            
            # save model if it got better loss
            current_loss = metrics['combined_val_loss']

            if current_loss < self.best_loss:
                print(f"New lowest loss: {current_loss}. Saving model...")
                self.best_loss = current_loss
                self.save_model(self.models[0])
            
        self.save_history()
