from __future__ import print_function

import pickle
import glob
import numpy as np
import os
import subprocess
import wfdb
import tqdm

DATA = "./"

def extract_wave(idx):
    """
    Reads .dat file and returns in numpy array. Assumes 2 channels.  The
    returned array is n x 3 where n is the number of samples. The first column
    is the sample number and the second two are the first and second channel
    respectively.
    """
    samp, ann = wfdb.rdsamp(idx)
    samp_no = np.arange(ann["sig_len"])
    data = np.concatenate([np.expand_dims(samp_no, axis=1), samp], axis=1)
    return data

def extract_annotation(idx):
    """
    The annotation file column names are:
        Time, Sample #, Type, Sub, Chan, Num, Aux
    The Aux is optional, it could be left empty. Type is the beat type and Aux
    is the transition label.
    """
    ann = wfdb.rdann(idx, extension="atr")
    labels = [ann.sample/ann.fs, ann.sample, ann.symbol, ann.subtype, ann.chan, ann.num, ann.aux_note]
    return labels

def extract(idx):
    """
    Extracts data and annotations from .dat and .atr files.
    Returns a numpy array for the data and a list of tuples for the labels.
    """
    data = extract_wave(idx)
    labels = extract_annotation(idx)
    return data, labels

def save(example, idx):
    """
    Saves data with numpy.save (load with numpy.load) and pickles labels. The
    files are saved in the same place as the raw data.
    """
    np.save(os.path.join(DATA, idx), example[0])
    with open(os.path.join(DATA, "{}.pkl".format(idx)), 'wb') as fid:
        pickle.dump(example[1], fid)

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA, "*.dat"))
    idxs = [os.path.basename(f).split(".")[0] for f in files]
    for idx in tqdm.tqdm(idxs):
        example = extract(idx)
        save(example, idx)
        tqdm.tqdm.write("Example {}".format(idx))

