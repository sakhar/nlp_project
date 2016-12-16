__author__ = 'sakhar'
#import sklearn
import pickle
from sklearn.metrics.classification import cohen_kappa_score
import numpy as np
from nltk.metrics.agreement import AnnotationTask

def normalize(labels):
    new_labels = []
    for l in labels:
        if l == 2:
            new_labels.append(1)
        elif l==4:
            new_labels.append(5)
        else:
            new_labels.append(l)
    return new_labels

def average_label(labels):
    l = [value for value in labels if value !=6]
    if not l:
        return 6
    try:
        return int(round(np.mean(l)))
    except:
        print labels,l
        return []

#print sklearn.__version__
#movie_name = 'godfather'
#movie_name = 'inception'
movie_name = 'shawshank'

scenes = pickle.load(open(movie_name+'_scenes.p','rb'))

print movie_name
pairs=0

all_characters = []
all_persons = []
for scene in scenes:
    all_persons.extend(scene.persons)
    all_characters.extend(scene.characters)

print 'len(scenes):', len(scenes)
print 'len(set(all_characters)):', len(set(all_characters))
print 'len(set(all_persons))', len(set(all_persons))
print 'len(set(all_persons).union(all_characters))', len(set(all_persons).union(all_characters))


labels = pickle.load(open('labels_tian/'+movie_name+'_pair_labels.p','rb'))
labels2 = pickle.load(open('labels_anant/'+movie_name+'_pair_labels.p','rb'))
labels3 = pickle.load(open('labels_avery/'+movie_name+'_pair_labels.p','rb'))


l1 = []
for v in labels.values():
    l1.extend(v.values())

l2 = []
for v in labels2.values():
    l2.extend(v.values())

l3 = []
for v in labels3.values():
    l3.extend(v.values())

#l1 = [1 if l == 2 elif 5 if l == 4 else l for l in l1]

print cohen_kappa_score(l1,l2)
print cohen_kappa_score(l1,l3)
print cohen_kappa_score(l2,l3)
print '==============='
print cohen_kappa_score(normalize(l1),normalize(l2))
print cohen_kappa_score(normalize(l1),normalize(l3))
print cohen_kappa_score(normalize(l2),normalize(l3))

pairs_labels = {}
pairs_labels2 = {}
pairs_labels3 = {}

for i in range(len(scenes)):
    scene = scenes[i]
    #print scene.texts
    for j in range(len(scene.pairs)):
        pair = scene.pairs[j]
        if pair not in pairs_labels:
            pairs_labels[pair] = []
            pairs_labels2[pair] = []
            pairs_labels3[pair] = []
        try:
            pairs_labels[pair].append(labels[i][j])
            pairs_labels2[pair].append(labels2[i][j])
            pairs_labels3[pair].append(labels3[i][j])
        except:
            #print 'err:', i, j
            pass

avg1 = {}
avg2 = {}
avg3 = {}
for pair in pairs_labels:
    avg1[pair] = average_label(pairs_labels[pair])
    avg2[pair] = average_label(pairs_labels2[pair])
    avg3[pair] = average_label(pairs_labels3[pair])


a=cohen_kappa_score(avg1.values(),avg2.values())
b=cohen_kappa_score(avg1.values(),avg3.values())
c=cohen_kappa_score(avg2.values(),avg3.values())

print '---- AVG ----'
print a
print b
print c
print (a+b+c)/3
print '==============='
a=cohen_kappa_score(normalize(avg1.values()),normalize(avg2.values()))
b=cohen_kappa_score(normalize(avg1.values()),normalize(avg3.values()))
c=cohen_kappa_score(normalize(avg2.values()),normalize(avg3.values()))
print cohen_kappa_score(normalize(avg1.values()),normalize(avg2.values()))
print cohen_kappa_score(normalize(avg1.values()),normalize(avg3.values()))
print cohen_kappa_score(normalize(avg2.values()),normalize(avg3.values()))
print (a+b+c)/3