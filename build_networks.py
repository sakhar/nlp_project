__author__ = 'sakhar'
from parsers import parse_godfather
import networkx as nx
import matplotlib.pyplot as plt
import pickle



def get_nets(movie_name):
    #############
    # please change the movie name
    #movie_name = 'godfather'
    #movie_name = 'shawshank'
    #movie_name = 'inception'

    scenes = pickle.load(open(movie_name+'_scenes.p','rb'))



    #print len(scenes)

    scenes_graph = nx.DiGraph()


    for i in range(len(scenes)):
        '''
        print '#####'
        print 'scene:', i
        print 'characters in scene:', scenes[i].characters
        for text in scenes[i].texts:
            print text
        '''
        for character in scenes[i].characters:
            scenes_graph.add_edge('C_'+character,'S_'+str(i))

        for person in set(scenes[i].persons).union(set(scenes[i].characters)):
        #for person in scenes[i].persons:
            scenes_graph.add_edge('S_'+str(i),'C_'+person)

        #print '#####'
#sentiment_score 2.63676234054 0.0360815740937
    X = []
    Y = []

    for node in scenes_graph:
        if 'S_' in node:
            X.append(node)
        else:
            Y.append(node)

    X = set(X)
    Y = set(Y)

    #print len(X)
    #print len(Y)

    pos = dict()
    pos.update( (n, (1, i*2)) for i, n in enumerate(X) ) # put nodes from X at x=1
    pos.update( (n, (2, i*2)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    #nx.draw(scenes_graph, pos=pos,with_labels = True)
    #plt.show()

    scene_nodes = [node for node in scenes_graph.node if 'S_' in node]

    characters_graph = nx.DiGraph()

    for node in scene_nodes:
        successors = scenes_graph.successors(node)
        predecessors = scenes_graph.predecessors(node)
        #print successors
        if 'C_ELDERLY JAPANESE MAN' in successors or 'C_ELDERLY JAPANESE MAN' in predecessors:
            print 'found in:', node
            print successors
            print predecessors
        for predecessor in predecessors:
            for successor in successors:
                if successor == predecessor:
                    continue
                if characters_graph.has_edge(predecessor,successor):
                    characters_graph[predecessor][successor]['weight'] += 1
                else:
                    characters_graph.add_edge(predecessor,successor,weight=1)


    #print 'len:', len(characters_graph.edges())
    #nx.write_gexf(scenes_graph, movie_name+"-unsigned.gexf")

    #nx.write_gexf(characters_graph, movie_name+"-character-unsigned.gexf")

    return characters_graph,scenes_graph