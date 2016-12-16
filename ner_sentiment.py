__author__ = 'sakhar'
import parsers
from pycorenlp import StanfordCoreNLP
import json
import pickle

nlp = StanfordCoreNLP('http://localhost:9000')

scenes = parsers.parse_inception()
#scenes = parsers.parse_godfather()
#scenes = parsers.parse_godfather2()
#scenes = parsers.parse_shawshank()
#scenes = parsers.parse_darkknight()

#print scenes[1].desc
def get_sentiments(text):
    sentiments = []
    sentiment_out = nlp.annotate(text,properties={'annotators':'sentiment', 'timeout': '20000'})
    sentences = json.loads(sentiment_out)['sentences']

    for sentence in sentences:
        words = [token['word'] for token in sentence['tokens']]
        #print ' '.join(words), sentence['sentiment'], 5-int(sentence['sentimentValue'])
        #print
        sentiments.append(1+int(sentence['sentimentValue']))
    return sentiments

for scene in scenes:
    #print scene.desc
    output = nlp.annotate(scene.desc,properties={'annotators':'ner', 'timeout': '20000'})
    out = json.loads(output)
    ners = []
    sentences_tokens = [sentence['tokens'] for sentence in out['sentences']]
    persons = []

    for text in scene.texts:
        try:
            scene.add_sentiment(get_sentiments(text[1]))
        except:
            print 'err:', text
            print unicode(text)

    for sentence in out['sentences']:
        words = [token['word'] for token in sentence['tokens']]
        ners = [token['ner'] for token in sentence['tokens']]


        while words:
            try:
                idx = ners.index('PERSON')

                endIdx = idx
                while(ners[endIdx]=='PERSON'):
                    endIdx += 1

                #print words[idx:endIdx]
                #print ners[idx:endIdx]
                persons.append(' '.join(words[idx:endIdx]).upper())

                words = words[endIdx:]
                ners = ners[endIdx:]


                #print words
                #print ners
            except:
                break
    for person in persons:
        scene.add_person(person)


for scene in scenes:
    print 'characters:', scene.characters
    print 'persons:', scene.persons
    for i in range(len(scene.texts)):
        print scene.texts[i]
        print scene.sentiments[i]

    print
    print


all_characters = []
all_persons = []
for scene in scenes:
    all_persons.extend(scene.persons)
    all_characters.extend(scene.characters)


for scene in scenes:
    all_chars = [person for person in scene.persons if person in all_characters]
    all_chars.extend(scene.characters)
    all_chars = list(set(all_chars))

    for char in all_chars:
        if char not in scene.characters:
            continue
        for char2 in all_chars:
            if char!=char2:
                scene.add_pair((char,char2))


#pickle.dump(scenes,open('godfather_scenes.p','wb'))
pickle.dump(scenes,open('inception_scenes.p','wb'))