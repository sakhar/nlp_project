__author__ = 'sakhar'

import pickle
import os
import datetime
from parsers import parse_godfather



#############
# please change the movie name + parser
movie_name = 'godfather'
# Change this as well
scenes = parse_godfather()


# key: sid: scene id
# value: {} dictionary, key: text id
#                       value: label: 1-5: 1 very negative, 2: negative, 3: neutral, 4: positive, 5: very positive

# {sid0:{tid0:label,tid1:label, ..., tid_n:label}, sid1:{...}, ...}

try:
    labels = pickle.load(open(movie_name+'_labels.p','rb'))
except:
    labels = {}

print 'Done:', len(labels), 'scenes out of', len(scenes)

print labels


# to automatically do backups
def backup(labels):
    last_length = 0

    if not os.path.exists("backups"):
        os.makedirs("backups")

    for file in os.listdir("backups"):
        if file.endswith(".backup"):
            f = pickle.load(open('backups/'+file,'rb'))
            if len(f) > last_length:
                last_length = len(f)
    print '### len(labels):', len(labels)

    # change this if you like,
    n_scenes = 2
    if (len(labels)-last_length) > n_scenes:
        time_stamp = str(datetime.datetime.now())
        pickle.dump((labels),open('backups/'+time_stamp+'.backup','wb'))




print '# scenes:', len(scenes)


print 'You will be annotating a lot of text! First you will see all texts in a scene,\n' \
      'then you will label each text in that scene! Good luck!'

ex = False

for i in range(len(scenes)):
    print '#####'
    print 'scene:', i
    print 'characters in scene:', scenes[i].characters
    if len(scenes[i].texts) == 0:
        print 'This scene is empty; skipped'
        continue
    if i in labels:
        if len(labels[i]) == len(scenes[i].texts):
            print 'This scene is already labeled; skipped'
            continue

    for j in range(len(scenes[i].texts)):
        uid = (i,j)
        text = scenes[i].texts[j]
        print uid, text
    print '$ END of scene '+str(i)


    # labeling

    skip = False

    print
    print
    print 'For each question, please answer with a number between 1-6; 1: Very Negative, 2: Negative, 3: Neutral, 4: Positive' \
          ', 5: Very Positive, 6: Cannot determine'
    print '0 to skip this scene, -1 to exit the program and continue later'
    print '(sceneid, text id)'
    print
    print
    labels[i] = {}

    for j in range(len(scenes[i].texts)):
        uid = (i,j)
        while True:
            try:
                text = scenes[i].texts[j]
                print uid, text
                answer = raw_input(str(uid)+' label: ')
                label = int(answer)
                if label == -1:
                    ex = True
                    break
                elif label == 0:
                    skip = True
                    break
                if not label > 0 or not label < 7:
                    raise
                break
            except:
                pass
        if skip:
            break
        if ex:
            exit()
        labels[i][j] = label
    print '#####'



    pickle.dump(labels,open(movie_name+'_labels.p','wb'))

    backup(labels)

    print
    print 'You might clear the screen now (recommended) cmd + k for Mac'

    while True:
        try:
            answer = raw_input('enter any number to continue, e to exit:')
            if answer.lower() == 'e':
                ex = True
                break
            int(answer)
            break
        except:
            pass
    if ex:
        exit()

print 'Glad you are done!'
print 'Done:', len(labels), 'threads out of', len(scenes)

