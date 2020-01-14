# %%
# coding: utf-8

# # Notebook for data preparation
# 
# Concat all datasets [mitdb, FH-2] and set samplerate to 360hz. Exports X.npy and Y.npy

# %%
import wfdb
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.signal import resample
import scipy.signal as sig


# %%
one_hot_af = [0, 1]
one_hot_sr = [1, 0]

# %%
dataset_timhf   = 0
dataset_ecgsyn  = 1
dataset_2017    = 2
dataset_mitafib = 3

# %% [markdown]
# ## TIM-HF1
# https://gitlab.hpi.de/osm/mpws2019ap1/general

# %%
base_path = '../data/ecg/'
af_path = base_path + 'af'
sr_path = base_path + 'sr'

def readData(path):
    numpy_vars = []
    metadata_timhf = []

    for np_name in glob.glob(os.path.abspath(path) + '/*.npy'):
        try:
            # print("Reading file", np_name)
            rhythm_type = np_name.split("\\")[-2]
            metadata_timhf.append([dataset_timhf, rhythm_type + "-" + os.path.basename(np_name).split(".")[0]])
            # numpy_vars.append(resample(np.load(np_name, allow_pickle=True)[0], num=120*360))
        except:
            print("Could not load", np_name)
    return numpy_vars, metadata_timhf

af_raw_data, meta_af = readData(af_path)
sr_raw_data, meta_sr = readData(sr_path)

meta_tim = meta_af + meta_sr


# %%
X = af_raw_data + sr_raw_data
Y = [one_hot_af] * len(af_raw_data)
Y += [one_hot_sr] * len(sr_raw_data)

len(X)

# %% [markdown]
# ## EKG generator
# https://physionet.org/content/ecgsyn/1.0.0/

# %%
#.\ecgsyn.exe -s 360 -S 360 -n 160 -h 80
generated_path = base_path + "generated/ecgsyn.dat"

with open(generated_path) as f:
    content = f.readlines()

records = np.array([np.array(x.split()[:2], dtype=float) for x in content])[:120 * 360]

# plt.plot(records[:1000,0], records[:1000, 1])
# plt.show()

# %% [markdown]
# ## 2017 Dataset
# https://www.physionet.org/content/challenge-2017/1.0.0/#files

# %%
import wfdb

base_path_physionet = os.path.join("../data/training2017/")
record_path_pyhsionet = os.path.join(base_path_physionet, "RECORDS")
reference_path_physionet = os.path.join(base_path_physionet, "REFERENCE.csv")

with open(record_path_pyhsionet) as f:
    content = f.readlines()

records = [x.strip() for x in content] 

data_physionet = []
for entry in records:
    ecg_file = wfdb.rdsamp(os.path.join(base_path_physionet, entry ))
    data_physionet.append(resample(ecg_file[0], num=int(len(ecg_file[0]) * 1.2)).flatten())

# %%
slice_length = 120 * 360;

scaled_data = []
for data_slice in data_physionet:
    max_rand = slice_length - len(data_slice)
    pos = int(np.random.uniform(low=0, high=max_rand))
        
    line = np.concatenate([np.zeros(pos), data_slice, np.zeros((slice_length -pos - len(data_slice)))])
    
    scaled_data.append(line)

# %%
with open(reference_path_physionet) as f:
    reference = f.readlines()

finished_data = []
for line in range(len(scaled_data)):
    rhythm = (str(reference[line]).split(","))[1]
            
    if rhythm.strip() == "A":
        finished_data.append([scaled_data[line], one_hot_af])
    elif rhythm.strip() == "N":
        finished_data.append([scaled_data[line], one_hot_sr])
    else:
        # different classification
        pass

# %%
X_2017 = list(map(lambda x: x[0], finished_data))
Y_2017 = list(map(lambda x: x[1], finished_data))

# np.save("X_2017", X_2017)
# np.save("Y_2017", Y_2017)
X += X_2017
Y += Y_2017

len(X)

# %%
meta_2017 = list(map(lambda x: [dataset_2017, x], records))

# %% [markdown]
# ## MIT Arrhytmia
# https://physionet.org/content/mitdb/1.0.0/

# %%
base_path = os.path.join("../data/mit-bih-arrhythmia/")
data_path = os.path.join(base_path, "100")
record_path = os.path.join(base_path, "RECORDS")

with open(record_path) as f:
    content = f.readlines()

records = [x.strip() for x in content] 


# %%
arrhythmia = "(AFIB"
normal = "(N"

# categorize as arrythmia if more than 30 seconds AFIB
def is_arrhythmia(frame):
    sum = (frame == arrhythmia).sum()
    #result = (sum > len(frame) / 4) == True
    #return one_hot_af if result else one_hot_sr
    
    return int((sum > len(frame) / 4) == True)
    
def prepare_patient(patient_id, path=base_path):
    patient_path = os.path.join(path, patient_id)
    print(path, patient_id)
    
    annotations = wfdb.rdann(patient_path, extension="atr")
    sample, meta = wfdb.rdsamp(patient_path)
    
    sample_duration = len(sample) / annotations.fs / 60;
    dest_duration = 2;
    
    num_cuts = int(sample_duration / dest_duration)
    slicePoints = np.arange(1, num_cuts, dtype=int) * (annotations.fs * 120)
    
    sample_cuts = np.array_split(sample[:, 0], slicePoints)
    
    cut_length = len(sample_cuts[0])
    #cut_annotations = []
    cut_indizes = []

    sample_indizes = annotations.sample
    for index in range(15):
        min_index, max_index = index * cut_length, (index + 1) * cut_length
        indizes = np.where(np.logical_and(sample_indizes >= min_index, sample_indizes < max_index))
        cut_indizes.append(indizes)

    annotated_data = np.array(list(zip(sample_cuts, cut_indizes)))

    aux_notes = np.array(annotations.aux_note)
    
    cur_type = aux_notes[0] or normal;
    all_notes = []
    for index in range(len(annotated_data)):
        note = list(aux_notes[annotated_data[index][1]])
        
        for y_index in range(len(note)):
            if note[y_index] == "":
                note[y_index] = cur_type
            else:
                cur_type = note[y_index]
        all_notes.append(np.append(annotated_data[index], is_arrhythmia(np.array(note))))
    
    return all_notes[:-1]


# %%
patients = []
for record in records:
    # print("Preparing patient:", record, "\n")
    
    patients.append(prepare_patient(record, base_path))

arrhythmia_slices = []
normal_slices = []

for patient in patients:
    for frame in patient:
        if frame[2] == 1:
            arrhythmia_slices.append([frame[0].tolist(), one_hot_af])
        else:
            normal_slices.append([frame[0].tolist(), one_hot_sr])
        


# %%
len(normal_slices)

# %%
all_slices = np.array(arrhythmia_slices + normal_slices)

X_MIT = all_slices[:, 0].tolist()
Y_MIT = all_slices[:, 1].tolist()


# %%
X += X_MIT
Y += Y_MIT

# %%
meta_mit = list(map(lambda x: [dataset_mitafib, x], records))


# %% [markdown]
# ## Data augmentation

# %%
# Baseline drift

# %%
# replace a given range with zeros to simulate errors during measurement
def addRandomZero(blob, ratio=0.1):
    data = np.array(blob)
    replace_len = int(len(blob) * ratio)
    start = np.random.randint(len(blob) - replace_len, size=1)[0]
    
    indizes = np.arange(start, start + replace_len + 1)
    np.put(data, indizes, 0)
    
    return data


# %%
# add noise to data
def addNoise(blob, std_dev=0.05):
    data = np.array(blob)
    noise = np.random.normal(0,std_dev,len(data))
    
    return data + noise


# %%
Z = np.array(meta_tim + meta_2017 + meta_mit)

# %%
np.save("X", X)
np.save("Y", Y)
np.save("Z", Z)

