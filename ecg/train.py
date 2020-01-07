# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import

import argparse
import json
import tensorflow.keras
import numpy as np
import os
import random
import time

import network
import load
import util

MAX_EPOCHS = 100

def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")

def train(args, params):

    print(f"Loading training set...{params['train']}")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")


    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })

    model = network.build_network(**params)

    stopping = tensorflow.keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    checkpointer = tensorflow.keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)

    batch_size = params.get("batch_size", 32)

    if args.multi_gpu:
        from tensorflow.keras.utils import multi_gpu_model
        #from tensorflow.keras import backend
        from tensorflow.python.client import device_lib

        avail_gpus = len([x for x in device_lib.list_local_devices() if x.device_type  == "GPU"])
        if not isinstance(args.multi_gpu, bool):
            num_gpus = min(args.multi_gpu, avail_gpus)
        else:
            num_gpus = avail_gpus

        #num_gpus = len(backend.tensorflow_backend._get_available_gpus())
        batch_size *= num_gpus
        print(f"training on {num_gpus} gpus with batch_size {batch_size}, μ-batch™_size {batch_size//num_gpus}")
        parallel_model = multi_gpu_model(model, gpus=num_gpus)
        network.add_compile(parallel_model,**params)
        model = parallel_model

    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    parser.add_argument("--multi-gpu", "-m", help="enable multi gpu-training",
            action='store', default=False, const=True, nargs="?", type=int)
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    print(args.multi_gpu)
    train(args, params)
