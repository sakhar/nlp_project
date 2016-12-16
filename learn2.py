__author__ = 'sakhar'
import pickle
from build_networks import get_nets
import math
import random
from collections import Counter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score,precision_recall_fscore_support,recall_score,f1_score
import sklearn
from sklearn import svm
import networkx as nx
import sklearn.feature_selection
import nltk

normalize = False

# majority vote
def majority(ll):
    ll_norm = []
    for l in ll:
        if l == 1 or l ==2:
            ll_norm.append(1)
        elif l==4 or l==5:
            ll_norm.append(5)
        else:
            ll_norm.append(l)
    if not normalize:
        ll_norm = ll

    l = [value for value in ll_norm if value !=6]

    random.shuffle(l)

    common = Counter(l).most_common()
    ties = []

    for num in common:
        if num[1] == common[0][1]:
            ties.append(num[0])
        else:
            break

    if len(ties) == 0:
        #return []
        return 6

    #return int(math.floor(np.mean(ties)))

    final_label = round(np.mean(ties))

    if not normalize:
        return final_label

    if final_label == 1 or final_label ==2:
        return 1
    elif final_label == 4 or final_label ==5:
        return 5
    else:
        return final_label
    #return round(np.mean(ties))

def average_label(labels):
    l = [value for value in labels if value !=6]
    if not l:
        return []
    try:
        final_label = int(round(np.mean(l)))
        if not normalize:
            return final_label
        if final_label == 1 or final_label ==2:
            return 1
        elif final_label == 4 or final_label ==5:
            return 5
        else:
            return final_label
        #return int(round(np.mean(l)))
    except:
        print labels,l
        return []

movie_name = 'godfather'
#movie_name = 'inception'
#movie_name = 'shawshank'
scenes = pickle.load(open(movie_name+'_scenes.p','rb'))

labels = pickle.load(open('labels_tian/'+movie_name+'_pair_labels.p','rb'))
labels2 = pickle.load(open('labels_anant/'+movie_name+'_pair_labels.p','rb'))
labels3 = pickle.load(open('labels_avery/'+movie_name+'_pair_labels.p','rb'))

pairs_labels = {}

for i in range(len(scenes)):
    scene = scenes[i]
    #print scene.texts
    for j in range(len(scene.pairs)):
        pair = scene.pairs[j]
        if pair not in pairs_labels:
            pairs_labels[pair] = []

        scene_pair_labels = []
        try:
            scene_pair_labels.append(labels[i][j])
        except:
             print 'label1 error:', i,j
        try:
            scene_pair_labels.append(labels2[i][j])
            pass
        except:
            print 'label2 error:', i,j

        try:
            scene_pair_labels.append(labels3[i][j])
            pass
        except:
            print 'label2 error:', i,j


        #print pair, scene_pair_labels, majority(scene_pair_labels)
        pairs_labels[pair].append(majority(scene_pair_labels))


        try:
            pass
            #pairs_labels[pair].append(labels[i][j])

        except:
            print 'error:', i,j

#print labels
#print labels2
#exit()
print 'len(pairs_labels):', len(pairs_labels)
print 'len(scenes):', len(scenes)

characters_graph,scenes_graph = get_nets(movie_name)

undir_characters_graph = characters_graph.to_undirected()
undir_scenes_graph = scenes_graph.to_undirected()

scenes_eigen_centralities = nx.eigenvector_centrality_numpy(scenes_graph,weight='weight')
scenes_undir_eigen_centralities = nx.eigenvector_centrality_numpy(undir_scenes_graph,weight='weight')

characters_eigen_centralities = nx.eigenvector_centrality_numpy(characters_graph,weight='weight')
characters_undir_eigen_centralities = nx.eigenvector_centrality_numpy(undir_characters_graph,weight='weight')

def preprocessing(word, preprocess=True):
    try:
        if preprocess:
            word = word.lower()
            word = ''.join('8' if c.isdigit() else c for c in word)
    except:
        print 'error!!'
    return word


def get_features(char1, char2):
    features = {}

    pair = (char1,char2)
    count_relative_sentiment = {i:0 for i in range(1,6)}
    count_sentiment = {i:0 for i in range(1,6)}
    try:
        num_common_scenes = sum(1 for _ in nx.common_neighbors(undir_characters_graph,'C_'+char1,'C_'+char2))
    except:
        num_common_scenes = 0
        print 'error:', [char1], [char2]

    char1_scenes = undir_scenes_graph.degree('C_'+char1)
    char2_scenes = undir_scenes_graph.degree('C_'+char2)

    features.update({'NET_common_scenes':num_common_scenes})
    features.update({'NET_char1_scenes':char1_scenes,'NET_char2_scenes':char2_scenes})

    features.update({'NET_ch1_char_eigen_centrality':characters_eigen_centralities['C_'+char1]})
    features.update({'NET_ch2_char_eigen_centrality':characters_eigen_centralities['C_'+char2]})

    features.update({'NET_ch1_char_undir_eigen_centrality':characters_undir_eigen_centralities['C_'+char1]})
    features.update({'NET_ch2_char_undir_eigen_centrality':characters_undir_eigen_centralities['C_'+char2]})

    features.update({'NET_ch1_scene_eigen_centrality':scenes_eigen_centralities['C_'+char1]})
    features.update({'NET_ch2_scene_eigen_centrality':scenes_eigen_centralities['C_'+char2]})

    features.update({'NET_ch1_scene_undir_eigen_centrality':scenes_undir_eigen_centralities['C_'+char1]})
    features.update({'NET_ch2_scene_undir_eigen_centrality':scenes_undir_eigen_centralities['C_'+char2]})

    bow = {}
    c1_bow = {}
    c2_bow = {}

    for i in range(len(scenes)):
        scene = scenes[i]
        for j in range(len(scene.texts)):
            text = scene.texts[j]
            if char1 == text[0]:
            #if True:
                sents = nltk.sent_tokenize(text[1])
                for sent in sents:
                    try:
                        tokens = nltk.word_tokenize(sent)
                    except:
                        print sent

                    for token in tokens:

                        pre_token = preprocessing(token, True)
                        try:
                            c1_bow['NLP_' + pre_token]
                        except:
                            c1_bow['NLP_' + pre_token] = 0
                        c1_bow['NLP_' + pre_token] += 1
                        if pair in scene.pairs:
                            try:
                                c2_bow['NLP_' + pre_token]
                            except:
                                c2_bow['NLP_' + pre_token] = 0
                            c2_bow['NLP_' + pre_token] += 1
            if char1 == text[0]:
            #if True:
                for sent in scene.sentiments[j]:
                    count_sentiment[int(sent)] += 1
                    if pair in scene.pairs:
                        count_relative_sentiment[int(sent)] += 1
    ratio_bow = {}
    for f in bow:
        try:
            ratio_bow[f] = float(c2_bow[f])/c1_bow[f]
        except:
            c1_bow[f] = 0
            c2_bow[f] = 0
        #print ratio_bow[f], c2_bow[f],bow[f]

    #print count_sentiment, count_relative_sentiment
    #print char1_scenes, scenes_graph.degree('C_'+char1)
    #print char2_scenes, scenes_graph.degree('C_'+char2)

    overall = {i:0 if count_sentiment[i] == 0 else count_relative_sentiment[i]/float(count_sentiment[i]) for i in range(1,6)}
    x = 0.0
    y = 0.0
    for i in overall:
        x += i*overall[i]
        y += overall[i]


    ch1_tri = nx.triangles(undir_characters_graph,'C_'+char1)
    ch2_tri = nx.triangles(undir_characters_graph,'C_'+char2)

    features.update({'SENT_count_all_sentiment_'+str(i):count_sentiment[i] for i in count_sentiment})
    features.update({'SENT_count_relative_sentiment_'+str(i):count_relative_sentiment[i] for i in count_relative_sentiment})
    features.update({'SENT_ratio_sentiment_'+str(i):overall[i] for i in overall})
    features.update({'SENT_sentiment_score':x/y if y !=0 else 0})
    features.update({word+'_ratio':ratio_bow[word] for word in ratio_bow})
    features.update({word+'_c2_bow':c2_bow[word] for word in c2_bow})
    features.update({word+'_c1_bow':c1_bow[word] for word in c1_bow})
    #features.update(bow)


    try:
        num_common_neighbors = sum(1 for _ in nx.common_neighbors(undir_characters_graph,'C_'+char1,'C_'+char2))
    except:
        num_common_neighbors = 0
        print 'error:', [char1], [char2]
    features.update({'NET_common_neighbors': num_common_neighbors})
    features.update({'NET_ch1_tri': ch1_tri})
    features.update({'NET_ch2_tri': ch2_tri})
    try:
        features.update({'NET_common_neighbors/tri': 2.0*num_common_neighbors/float(ch1_tri+ch2_tri)})
    except:
        #features.update({'common_neighbors/tri': 2.0*num_common_neighbors/float(ch1_tri+ch2_tri)})
        features.update({'NET_common_neighbors/tri': 0})
    features.update({'NET_ch1_degree_cent':nx.degree_centrality(undir_characters_graph)['C_'+char1]})
    features.update({'NET_ch2_degree_cent':nx.degree_centrality(undir_characters_graph)['C_'+char2]})

    features.update({'NET_ch1_degree':nx.degree(undir_characters_graph)['C_'+char1]})
    features.update({'NET_ch2_degree':nx.degree(undir_characters_graph)['C_'+char2]})

    features.update({'NET_ch1_W_degree':nx.degree(undir_characters_graph,weight='weight')['C_'+char1]})
    features.update({'NET_ch2_W_degree':nx.degree(undir_characters_graph,weight='weight')['C_'+char2]})


    #print features

    return features

X = []
y = []

for pair in pairs_labels:
    #if len(pairs_labels[pair]) < 3:
    #    continue
    #print pair, pairs_labels[pair], label(pairs_labels[pair]), average_label(pairs_labels[pair])
    p_label = average_label(pairs_labels[pair])


    if not p_label:
        continue
    #print 'features:'
    features = get_features(pair[0],pair[1])
    X.append(features)
    y.append(p_label)



def classify(clf, X_dev, y_dev):
    accuracy = clf.score(X_dev, y_dev)*100
    y_pred = clf.predict(X_dev)

    f1_all = f1_score(y_dev, y_pred, pos_label=None,average='macro')
    recall = recall_score(y_dev, y_pred, pos_label=None,average='macro')
    precision = precision_score(y_dev, y_pred, pos_label=None,average='macro')

    print accuracy, recall, precision, f1_all
    print precision_recall_fscore_support(y_dev, y_pred, pos_label=None,average='weighted')

    for l, p in zip(y_dev,y_pred):
        print l,p

    return y_pred


dictVec = DictVectorizer(sparse=False)


X_vectors = dictVec.fit_transform(X)

X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_vectors,y,
                                                                test_size=0.25, random_state=1988)


F, pval = sklearn.feature_selection.f_classif(np.asanyarray(X_train),y_train)

my_dict2 = {y:x for x,y in dictVec.vocabulary_.items()}
good_features=[]
for f, p, i in zip(F, pval,my_dict2):

    #if p <= 0.05 or 'NLP_' not in my_dict2[i]:
    #if p <= 0.05 or 'NLP_' not in my_dict2[i]:
    #if 'SENT_' in my_dict2[i]:
    #if 'NET_' in my_dict2[i]:
    #if 'SENT_' in my_dict2[i] or (p <= 0.05 and 'NLP_' in my_dict2[i]) :
    #if 'SENT_' in my_dict2[i] or 'NET_' in my_dict2[i]:
    #if p <= 0.05 and 'NLP_' in my_dict2[i]:
    #if p <= 0.05 and 'NLP_' in my_dict2[i]:
    if p <= 0.05 or 'NLP_' not in my_dict2[i]:
        print my_dict2[i],f,p
        good_features += [my_dict2[i]]

print len(good_features)

mask = [True if f in good_features else False for f in dictVec.get_feature_names()]

dictVec.restrict(mask)


X_vectors = dictVec.transform(X)
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_vectors,y, test_size=0.25, random_state=1988)

#X_vectors = dictVec.transform(X)

#X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_vectors,y, test_size=0.25, random_state=1988)



print 'X_train[0]:', len(X_train[0])


print 'X_train:', len(X_train)
print 'X_test:', len(X_test)

my_dict2 = {y:x for x,y in dictVec.vocabulary_.items()}
maxes = np.amax(X_train,axis=0)
mins = np.amin(X_train,axis=0)
means = np.mean(X_train,axis=0)
stds = np.std(X_train,axis=0)


for i in range(len(my_dict2)):
    if maxes[i]-mins[i] == 0:
        continue
    X_train[:,i] -= means[i]
    X_train[:,i] /= maxes[i]-mins[i]

    X_test[:,i] -= means[i]
    X_test[:,i] /= maxes[i]-mins[i]

class_weights = ['balanced',None]
C_range = [1,10,30,50,100,200]
gamma_range = np.logspace(-3, 3, 7)
param_grid = [{'kernel': ['linear'], 'C': C_range, 'class_weight': class_weights},
              {'kernel':['rbf'],'C': C_range, 'class_weight': class_weights, 'gamma': gamma_range}]

def cross_scores(classifier,X,y):
    y_pred = classifier.predict(X)

    f1_all = f1_score(y, y_pred, pos_label=None,average='weighted')
    #f1_all = f1_score(y, y_pred, pos_label=None,average='macro')
    #recall = recall_score(y, y_pred, pos_label=None,average='macro')
    #precision = precision_score(y, y_pred, pos_label=None,average='macro')

    return f1_all

print 'start grid search'
grid = GridSearchCV(svm.SVC(cache_size=3000), param_grid=param_grid,cv=3,scoring=cross_scores)
grid.fit(X_train, y_train)

print grid.best_params_

clf = grid.best_estimator_
clf.fit(X_train, y_train)

y_pred = classify(clf, X_test,y_test)

for i in range(10):
    print
print nx.triangles(undir_characters_graph)


pairs = []
for pair in pairs_labels:
    #if len(pairs_labels[pair]) < 3:
    #    continue
    #print pair, pairs_labels[pair], label(pairs_labels[pair]), average_label(pairs_labels[pair])
    p_label = average_label(pairs_labels[pair])
    if not p_label:
        continue
    pairs.append(pair)

pairs_train, pairs_test = sklearn.cross_validation.train_test_split(pairs ,test_size=0.25, random_state=1988)

map = {}
for node in characters_graph.node:
    map[node] = node[2:]
nx.relabel_nodes(characters_graph,map,copy=False)
print characters_graph.edge

for pair in pairs:
    p_label = average_label(pairs_labels[pair])
    characters_graph.edge[pair[0]][pair[1]]['true_label'] = str(p_label)

for pair,t_label, p_label in zip(pairs_test, y_test, y_pred):
    print pair,t_label, p_label
    characters_graph.edge[pair[0]][pair[1]]['predicted_label'] = str(p_label)
    if t_label == p_label:
        characters_graph.edge[pair[0]][pair[1]]['predicted_correctly'] = 'yes'
    else:
        characters_graph.edge[pair[0]][pair[1]]['predicted_correctly'] = 'no'

nx.write_gexf(characters_graph, movie_name+"-character-signed.gexf")

