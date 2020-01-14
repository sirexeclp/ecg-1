# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:43:22 2018

@author: ktvsp
"""

from enum import Enum, auto

class TimeUnit(Enum):
	MilliSecond = "ms"
	Sample = "samples"
	Second = "s"
	Minute = "min"
	Hour = "h"

class AmplitudeUnit(Enum):
	ADC = "adc"
	Volt = "V"
	MilliVolt = "mV"

class UnitConverter():
	"""Diese Klasse wandelt verschiedene Zeit-Einheiten ineinander um."""
	samplerate = 300

	@staticmethod    
	def samp2Sec( value):
		return value/UnitConverter.samplerate
	@staticmethod
	def samp2Min(value):
		return value/(UnitConverter.samplerate * 60)
	@staticmethod
	def samp2Hour( value):
		return value/(UnitConverter.samplerate * 3600)
	@staticmethod
	def samp2Ms(value):
		return UnitConverter.samp2Sec(value)*1000
	@staticmethod
	def sec2Samp( value):
		return value * UnitConverter.samplerate
	@staticmethod
	def min2Samp( value):
		return value * UnitConverter.samplerate * 60
	@staticmethod
	def hour2Samp( value):
		return value * UnitConverter.samplerate * 3600
	@staticmethod
	def ms2Samp(value):
		return UnitConverter.sec2Samp(value/1000)

	@staticmethod
	def convert(value,fromUnit=TimeUnit.Sample,toUnit=TimeUnit.Sample, sampleRate = None):
		"""Konvertiert den Wert von einer Einheit in eine andere.
		value = Wert der Konvertiert werden soll
		fromUnit = Ausgangs-Einheit
		toUnit = Ziel-Einheit
		sampleRate = verwendete Abtastrate
		returns: Konvertierter Wert"""
		if sampleRate:
			UnitConverter.samplerate = sampleRate
		valueInSamp = None
		if fromUnit is TimeUnit.Sample:
			valueInSamp = value
		elif fromUnit is TimeUnit.Second:
			valueInSamp = UnitConverter.sec2Samp(value)
		elif fromUnit is TimeUnit.Minute:
			valueInSamp = UnitConverter.min2Samp(value)
		elif fromUnit is TimeUnit.Hour:
			valueInSamp = UnitConverter.hour2Samp(value)
		elif fromUnit is TimeUnit.MilliSecond:
			valueInSamp = UnitConverter.ms2Samp(value)
		else:
			raise ValueError(f"Unrecognised Unit {type(fromUnit)} {fromUnit}")
		
		if toUnit is TimeUnit.Sample:
			return valueInSamp
		elif toUnit is TimeUnit.Second:
			return UnitConverter.samp2Sec(valueInSamp)
		elif toUnit is TimeUnit.Minute:
			return UnitConverter.samp2Min(valueInSamp)
		elif toUnit is TimeUnit.Hour:
			return UnitConverter.samp2Hour(valueInSamp)
		elif toUnit is TimeUnit.MilliSecond:
			return UnitConverter.samp2Ms(valueInSamp)
		else:
			raise ValueError(f"Unrecognised Unit {type(toUnit)} {toUnit}")
