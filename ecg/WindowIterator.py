# -*- coding: utf-8 -*-
from enum import Enum,auto
import numpy as np

class Padding(Enum):
	Zero = auto()
	#One = auto()
	Offset = auto()
	Same = auto()

class WindowIterator:
	"""Stellt eine gleitendes Fenster über Daten bereit.
	Das Fenster kann eine konstante oder variable Länge haben.
	Je nach Einstellungen sollte daher die Länge des Fensters mit 
	len() abgerufen werden."""
	
	def __init__(self,signal,windowLength,stride,padding, *args, **kwargs):
		"""Erstellt ein Fenster für das angegebene Signal.
		signal = das Signal über welches iteriert werden soll
		windowLength = die angestrebte Fensterlänge
		stride = der Fortschritt des Fensters mit jeder Iteration
		padding = Optionen für die Randbehandlung :
			-Zero = es werden '0' angehangen, konstante Fensterlänge und gleiche Länge von Einganssignal und Anzahl der Iterationen
			-Offset = das Fenster wird so platziert das es nicht über den Rand ragt, konstante Fensterlänge aber die Anzahl der Iterationen ist gerringer als die länge des Eingangssignals
			-Same = das Fenster wächst bzw. schrumpft am Rand, variable Fensterbreite aber gleiche Länge von Eingangssignal und anzahl der Iterationen"""
		assert(windowLength%2==1,"windowLength must be uneven")
		assert(len(signal)>windowLength,"windowLength must be less or equal to len(signal)")
		self.signal = signal
		self.position = 0
		self.windowLength = windowLength
		self.stride = stride
		self.padding = padding
	
	def __iter__(self):
		return self
	
	@property
	def halfWindow(self):
		return int(self.windowLength/2)
	
	def __next__(self):
		start = self.position - self.halfWindow
		end = self.position + self.halfWindow+1

		startOffset = 0
		endOffset = self.windowLength
	   
		if self.padding == Padding.Offset:
			start = self.position
			end = self.position + self.windowLength

		if start < 0:
			if self.padding == Padding.Same:
				start = 0
			elif self.padding == Padding.Zero:
				startOffset = - start
				start = 0
		
		if end > len(self.signal):
			if self.padding == Padding.Same:
				end = len(self.signal)
				if self.position >= end:
					raise StopIteration()
			elif self.padding == Padding.Offset:
					raise StopIteration()
			elif self.padding == Padding.Zero:
				endOffset = len(self.signal) -end
				end = len(self.signal)
				if self.position >= end:
					raise StopIteration()
		
		result = self.signal[start:end]
		if self.padding == Padding.Zero:
			tmp = np.zeros(self.windowLength)
			tmp[startOffset:endOffset] = np.array(result)
			result = tmp
			
		self.position += self.stride
		return result

