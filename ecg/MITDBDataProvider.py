# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:27:46 2018

@author: ktvsp
"""
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path

# +
#from rDetection.RDetector import ShannonEnergyHilbertDetector, diff3
from Signal import Signal
from UnitConverter import TimeUnit, UnitConverter
from typing import List
from tqdm import tqdm

STEP = 256

class MITDBDataProvider():
    DEFAULT_SAMPLERATE = 360
    """IO-Klasse zum Lesen von Aufnahmen und Annotationen der MIT-Datenbank."""

    def __init__(self):
        pass

    @staticmethod
    def readRecord(recordFile: str,data_root, online=False) -> Signal:
        """Liest die Aufnahme mit der ensprecheneden Nummer,
        falls online wird die Aufnahme aus dem Internet geladen.
        returns signal,annotation,samplerate"""
        
        db_path = data_root
        assert data_root.exists(), f"data root {data_root} not found!"

        #db_path = data_root / database_name
        db_record_names = (db_path / "RECORDS").read_text().splitlines()
        db_annotation_ext = (db_path / "ANNOTATORS").read_text().splitlines()[0].split("\t")[0]
        
        annotation = wfdb.rdann(recordFile, extension=db_annotation_ext)
        sampleRate = annotation.fs

        record = wfdb.rdsamp(recordFile)
        signal = record[0][:, 0]
        btuSig = Signal(signal, sampleRate, annotations=annotation)
        return btuSig

    @staticmethod
    def mostCommon(lst: List) -> str:
        """Bestimmt das am häufigsten vorkommende Element in der Liste"""
        return max(set(lst), key=lst.count)

    @staticmethod
    def calculateSlicePoints(sliceLength: int, totalLength: int):
        """Berechnet die Punkte, an denen die Aufnahmen zertrennt werden sollen.
        sliceLength = länge einer Teilaufnahme (sample)
        totalLength = länge der gesammten Aufnahme (sample)"""
        numberOfSlices = totalLength//sliceLength
        splitPoints = np.arange(1, numberOfSlices, dtype=int)*sliceLength
        return splitPoints

    @staticmethod
    def sliceSignal(signal: Signal, sliceLength: int = 60, new_sample_rate=DEFAULT_SAMPLERATE):
        """Zerteilt die Aufnahme in gleich Lange (default 60 Sekunden)
        lange Teile.
        signal = Signal Objekt, der Aufnahme
        sliceLength = länge der Teilaufnahmen in Sekunden
        returns: Tuple mit einer Liste der Teilaufnahmen und einer Liste
                der Stellen an denen gertrennt wurde """
        signal = signal.resample(new_sample_rate)
        UnitConverter.samplerate = signal.sampleRate
        splitPoints = MITDBDataProvider.calculateSlicePoints(
            sliceLength, len(signal.voltage))
        splitedSignal = np.split(signal.voltage, splitPoints)
        return (splitedSignal, splitPoints)

    @staticmethod
    def _unpackBeatAnnotations(signal: Signal):
        """'Entpackt' die Annotationen, für eine einfachere Weiterverarbeitung."""
        lastNote = ""
        for idx, note in enumerate(signal.beatAnnotations.aux_note):
            if "(" not in note:
                signal.beatAnnotations.aux_note[idx] = lastNote
            else:
                lastNote = note

    @staticmethod
    def sliceAnnotations(signal: Signal, splitPoints: List) -> List:
        """Zerteilt die Annotationen, passend zum zerteilten Signal.
        signal = Signal, das zerteilt werden soll
        splitPoints = Punkte an denen das Signal zerteilt wurde
        returns: Liste mit Annotationen für jedes Teil-Signal """
        MITDBDataProvider. _unpackBeatAnnotations(signal)
        rythmtypes = pd.DataFrame([(signal.beatAnnotations.sample[idx], x) for idx, x in enumerate(
            signal.beatAnnotations.aux_note) if x], columns=["sample", "value"])

        splitPoints = np.insert(splitPoints, 0, 0)
        splitPoints = np.append(splitPoints, len(signal.voltage))
        labels = []
        for begin, end in zip(splitPoints[:-1], splitPoints[1:]):
            tmp = []
            
            label_count = ((end - begin) // 256)
            last_label = list(rythmtypes[(rythmtypes["sample"] >= begin)
                                  & (rythmtypes["sample"] <= end)].value)[0]
            for l in range(label_count):
                new_lbl = list(rythmtypes[(rythmtypes["sample"] >= begin+(l*256))
                                  & (rythmtypes["sample"] <= begin+((l+1)*256))].value)
                if len(new_lbl) >0:
                    last_lbl = new_lbl[0]
                tmp.append(last_lbl)
            labels.append(tmp)
        return labels

    @staticmethod
    def detectRPeaksOnSlices(slices: List, sampleRate):
        """Erkennt R-Peaks auf den zerteilten Signalen."""
        sed = ShannonEnergyHilbertDetector(sampleRate)
        peaks = []
        rrs = []
        for s in slices:
            tmp = sed.detect(signal=s)
            peaks.append(tmp)
            rrs.append(diff3(tmp))
        return peaks, rrs

    @staticmethod
    def getRPeaksFromAnnotationOnSlices(signal: Signal, splitPoints: List):
        """Liest die R-Peaks aus den Annotationen der zerteilten Signale.
        signal = zu zerteilendes Signal
        splitPoints = Punkte an denen das Signal zerteilt wurde
        returns: Tuple mit Listen der RPeaks und RRIntervalle für jedes Teil-Signal"""
        peaks = []
        rrs = []
        annotations = MITDBDataProvider.filterNonBeatAnnotations(
            MITDBDataProvider.annotations2DataFrame(signal.beatAnnotations)
        )

        # annotations = np.array([signal.beatAnnotations.sample[idx] for idx \
        # 	, x in enumerate(signal.beatAnnotations.symbol) if x])

        splitPoints = np.insert(splitPoints, 0, 0)
        splitPoints = np.append(splitPoints, len(signal.voltage))
        for begin, end in zip(splitPoints[:-1], splitPoints[1:]):
            tmp = annotations[(annotations["sample"] >= begin)
                              & (annotations["sample"] <= end)]
            tmp = np.array(tmp["sample"], dtype=int)
            peaks.append(tmp)
            if len(tmp) > 2:
                rrs.append(diff3(tmp))
            else:
                print("empty")
                rrs.append([])
        return peaks, rrs

    @staticmethod
    def annotations2DataFrame(annotations):
        """Erstell einen Dataframe aus den Annotationen"""
        df = pd.DataFrame(np.array([np.array(annotations.sample), np.array(annotations.symbol), np.array(
            annotations.aux_note)]).T, index=annotations.sample, columns=["sample", "symbol", "aux_note"])
        df["sample"] = df["sample"].astype(int)
        return df

    @staticmethod
    def filterNonBeatAnnotations(dataFrame: pd.DataFrame) -> pd.DataFrame:
        """Entfernt Annotationen, die keine R-Zacken markieren."""
        assert(isinstance(dataFrame, pd.DataFrame))
        nonBeatAnnotations = [
            "[", "!", "]", "x", "(", ")", "p", "t", "u", "`", "'", "^", "|", "~", "+", "s", "T", "*", "D", "=", "\"", "@"]

        return dataFrame.loc[~dataFrame["symbol"].isin(nonBeatAnnotations)]


# -

# get_data_4_fed("/workspace/telemed5000/code/data/")


def load_database(database_name,data_root= "../data",record_list=None):
    data_root = Path(data_root)
    assert data_root.exists(), f"data root {data_root} not found!"
    
    db_path = data_root / database_name
    db_record_names = (db_path / "RECORDS").read_text().splitlines()
    db_annotation_ext = (db_path / "ANNOTATORS").read_text().splitlines()[0].split("\t")[0]
    db_records = []
    if record_list is not None:
        print(f"only loading records: {record_list}")
        db_record_names = record_list
    #print("loading records")
    for record_name in tqdm(db_record_names):
        record_path = db_path / record_name
        tmp_rec = MITDBDataProvider.readRecord(str(record_path),db_path)
        db_records.append(tmp_rec)
    return db_records


def get_one_hot_prepared(mitdb_rec, slice_len, num_classes = 4,fs=300):
    # mit_record_slices = [ MITDBDataProvider.sliceSignal(x,120-0.533333) for x in mitdb_rec]
    mit_record_slices = [ MITDBDataProvider.sliceSignal(x,slice_len,new_sample_rate=fs) for x in mitdb_rec]

    mit_annotation_slices = [ MITDBDataProvider.sliceAnnotations(x,y) for x,(_,y) 
                             in zip(mitdb_rec,mit_record_slices)]
    mit_annotation_slices = [[[label.strip("\x00")[1:] for label in labels] for labels in slize]
                             for slize in mit_annotation_slices]
    
    
    annotation_idx = {
        "AFIB":0
        ,"N":1
        ,"~":3
    }
    other = 2
    # 2 = other
    # 3 = noise
    
    mit_annotation_slices = [[[annotation_idx.get(x,other) for x in sli] for sli in record]
                             for record in mit_annotation_slices]
        
    from tensorflow.keras.utils import to_categorical
    num_classes = 4
    
    mit_annotation_slices = [[np.array([to_categorical(x, num_classes) for x in sli])\
                             for sli in rec] for rec in mit_annotation_slices]
    
    return mit_record_slices, mit_annotation_slices


def remove_short_slices(mit_record_slices, mit_annotation_slices, length):
    #length_int = UnitConverter.convert(length, fromUnit=TimeUnit.Second)
    #trunc_samp = STEP * (length_int // STEP)
    #mit_record_slices = []
    mit_record_slices = [[x for x in rec[0] if len(x)==length] for rec in mit_record_slices]
    
    mit_annotation_slices = [[y for x,y in zip(sli_x, sli_y) ]
                             for sli_x, sli_y in zip(mit_record_slices, mit_annotation_slices)]
    return mit_record_slices, mit_annotation_slices


def flatten_list(l):
    return [item for sublist in l for item in sublist]


from sklearn.model_selection import train_test_split
# Split the data
#x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.33, shuffle= True)
def slpit_slices(x,y,*args,**kwargs):
    splitted = [train_test_split(rec_x, rec_y,*args,**kwargs) 
                                 for rec_x, rec_y in zip(x, y)]
    x_train, x_test, y_train, y_test =  zip(*splitted)
    x_test = np.array(flatten_list(x_test))
    y_test = np.array(flatten_list(y_test))
    
    #x_train = [x[0] for x in splitted]
    #x_test = flatten_list([x[1] for x in splitted])
    #y_train = [x[2] for x in splitted]
    #y_test = flatten_list([x[3] for x in splitted])
    return x_train, x_test, y_train, y_test


def get_data_4_fed(data_path,test_size=0.33,shuffle=True, record_list=None):
    #
    mitdb_rec = load_database("mitdb", data_path, record_list=record_list)
    slice_len = 8960
    x, y = get_one_hot_prepared(mitdb_rec,slice_len=slice_len)
    
    x, y = remove_short_slices(x, y, slice_len)
    return slpit_slices(x,y, test_size=test_size,shuffle=shuffle,random_state=42)


def get_all_data_flattened(data_path,test_size=0.33,shuffle=True, record_list=None):
    pass
    #mitdb_rec = load_database("mitdb", data_path, record_list=record_list)
    #x, y = get_one_hot_prepared(mitdb_rec)
    #x, y = remove_short_slices(x, y)
    #return train_test_split(flatten_list(x), flatten_list(y),
    #                        test_size=test_size,shuffle=shuffle)


if __name__ == "__main__":
    records = ["201", "202", "203", "219", "222"]
    get_data_4_fed("/workspace/telemed5000/code/data/", record_list=records)


