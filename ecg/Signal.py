# -*- coding: utf-8 -*-
"""
Created on Sun May 27 23:00:36 2018

@author: ktvsp
"""
from enum import Enum, auto
import numpy as np
import jsonpickle
import matplotlib.pyplot as plt
from UnitConverter import UnitConverter, TimeUnit
#from ECGPlotter import ECGPlotter
import copy
import scipy.signal as sig

def diff3(x):
    """ calculates the discrete difference using a window of 3"""
    assert(len(x) > 2)
    x = np.array(x)
    dx = np.zeros(x.shape)
    dx[1:-1] = 0.5*(x[2:]-x[:-2])
    dx[[0, -1]] = dx[[1, -2]]
    return dx


def curryDict(func, var):
    """currys function func with variable var"""
    y = var

    def f(x):
        return func(x, **y)
    return f

class RhythmType(Enum):
    AFIB = auto()
    AFL = auto()
    BII = auto()
    N = auto()
    Test = auto()
    BadQuality = auto()
    Other = auto()

class Gender(Enum):
    Male = auto()
    Female = auto()

class Signal():
    """Speichert EKG-Daten und Annotationen"""

    def __init__(self, voltage, sampleRate, sampleNumber=None, millis=None, x=None, y=None, z=None, annotations=None,
                 recordID=None, gender=None, age=None, timestamp=None, rhythmType=None):

        self.voltage = np.array(voltage)
        self.sampleNumber = np.array(sampleNumber)
        self.millis = np.array(millis)
        self.accX = np.array(x)
        self.accY = np.array(y)
        self.accZ = np.array(z)
        self.recordID = recordID
        self.gender = gender
        self.age = age

        self.sampleRate = sampleRate
        self.beatAnnotations = annotations
        self.timestamp = timestamp
        self.rhythmType = rhythmType

    def resample(self, targetSampleRate):
        newLen = int(len(self.voltage)*(targetSampleRate/self.sampleRate))
        result = copy.deepcopy(self)
        result.sampleRate = targetSampleRate
        result.voltage = sig.resample(self.voltage, newLen)

        # try:
        # 	result.accX = sig.resample(self.accX, newLen)
        # 	result.accY = sig.resample(self.accY, newLen)
        # 	result.accZ = sig.resample(self.accZ, newLen)
        # except:
        # 	pass

        return result

    def saveJSON(self, fileName):
        """Serialisiert das Objekt als Json und speichert es in der angegebenen Datei."""
        jsonString = jsonpickle.encode(self)
        with open(fileName, "w") as file:
            file.write(jsonString)

    def fromJSON(self, fileName):
        """Deserialisiert das Objekt aus der angegebenen JSON-Datei und gibt es zurück."""
        jsonString = str()
        with open(fileName, "r") as file:
            jsonString = file.read()
        return jsonpickle.decode(jsonString)

    def analyse(self, rDetector):
        """Führt den R-Detektor auf dem EKG-Signal aus und gibt ein Analysiertes Signal zurück."""
        rDetector.sampleRate = self.sampleRate
        rPeaks = rDetector.detect(signal=self.voltage)
        aSignal = AnalysedSignal(self, rPeaks)
        return aSignal

    def plot(self, axis=None):
        """"Nutzt die ECGPlotter-Klasse um das Signal zu plotten."""
        return ECGPlotter.plotRawSignal(self.voltage, TimeUnit.Second, axis=axis)


class AnalysedSignal():

    def __init__(self, signal, rPeaks):
        self.signal = signal
        self.rPeaks = rPeaks

    @property
    def rrIntervals(self):
        """Berechnet die RR-Intervalle aus den R-Peaks."""
        return diff3(self.rPeaks)

    @property
    def heartRate(self):
        """
        Calculates the momentary hear rate based on rr intervalls in sec^-1
        """
        return 1/UnitConverter.samp2Min(self.rrIntervals)

    def plotHeartrate(self, axis=None):
        """Plotet die Herzfrequenz"""
        return ECGPlotter.plotHeartrate(self.heartRate, self.rPeaks, unit=TimeUnit.Second, axis=axis)

    def plotPoincare(self, scatter=False, removeOutliers=False, axis=None, title="Poincaré Plot"):
        """Erstellt einen Poincare-Plot der RR-Invervalle
        scatter = wenn True, werden Punkte gezeichnet und nicht mit Linien verbunden
        removeOutliers = wenn True, werden Ausreißer vorher entfernt
        axis = subplot-Axis
        title = Titel des Plots
        returns: subplot der erstell wurde"""
        intervals = UnitConverter.samp2Sec(self.rrIntervals)
        if removeOutliers:
            intervals = AnalysedSignal.removeOutliers(intervals)
        return ECGPlotter.scatterPlot(intervals, title, [0, 1], scatter, axis=axis)

    @staticmethod
    def removeOutliers(data, percent=0.05):
        """Entfernt ausreißer von den RR-Intervallen"""
        percentil = int(percent * len(data))
        deltaSorted = sorted([(item, index) for index, item in enumerate(data)])[
            percentil:-percentil]

        deltaFiltered = sorted(deltaSorted, key=lambda x: x[1])
        return [item for item, index in deltaFiltered]

    def plotLorenz(self, scatter=False, remOutl=False, axis=None, title="Lorenz Plot"):
        """Erstellt einen Lorenzplot der RR-Intervalle (s. auch plotPoincare)"""
        deltaRR = UnitConverter.samp2Sec(diff3(self.rrIntervals))
        if remOutl:
            deltaRR = AnalysedSignal.removeOutliers(deltaRR)
        ECGPlotter.scatterPlot(deltaRR, title, [-0.5, 0.5], scatter, axis=axis)

    def plot(self, axis=None):
        """Plottet das EKG mit erkannten R-Peaks als Annotationen."""
        ECGPlotter.plotAnnotatedSignal(
            self.signal.voltage, self.rPeaks, TimeUnit.Second, axis=axis)
