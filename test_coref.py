__author__ = 'sakhar'
import parsers
from pycorenlp import StanfordCoreNLP
import json
import pickle

nlp = StanfordCoreNLP('http://localhost:9000')

scenes = parsers.parse_godfather()

#dcoref