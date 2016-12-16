__author__ = 'sakhar'
from datastructure import scene
#from BeautifulSoup import BeautifulSoup
import re

def parse_godfather():
    script = 'godfather.txt'

    raw_text = open(script,'r')
    #cleantext = BeautifulSoup(raw_text).getText()
    lines = raw_text.read().split('\n')

    scenes = []

    # add the first scene
    scenes.append(scene())
    character_speaking = False
    comment_start = False
    desc = ''

    for line in lines:

        if character_speaking:
            #print current_character
            #print line

            if comment_start:
                if ')' in line:
                    comment_start = False
                continue
            # New Comment
            elif line[:4] == '\t\t\t(':
                if ')' not in line:
                    comment_start = True
                #print 'comment:' ,[line]
                continue

            # Speaker ends
            elif line == '':
                #print 'Speaker ends;', current_character
                #print current_text
                scenes[-1].add_text(current_character,current_text)

                current_character = ''
                character_speaking = False
                current_text = ''
                #print 'Speaker ends;', [line]

            else:
                current_text += line.strip() + ' '
                #print 'else print:', [line[2:]]
        elif '----DISSOLVE' in line:
                continue

        # Character speaking
        elif line[:3] == '\t\t\t':

            character_speaking = True
            current_text = ''

            current_character = line.strip()
            try:
                idx = current_character.index('(')
                current_character = current_character[:idx]
            except:
                pass
            current_character = current_character.strip()

            #print '[current_character]:', [current_character]


        # Scene ENDS
        elif line[:4] == 'EXT ' or line[:4] == 'INT ':

            scenes[-1].desc = desc
            desc = line[4:].strip()
            scenes.append(scene())

        # description
        else:
            desc += line.strip() + ' '
    #print lines
    #print len(scenes)

    return scenes

def parse_godfather2():
    script = 'godfather2.txt'

    raw_text = open(script,'r')
    #cleantext = BeautifulSoup(raw_text).getText()
    lines = raw_text.read().split('\n')

    scenes = []

    # add the first scene
    scenes.append(scene())
    character_speaking = False
    comment_start = False
    desc = ''

    for line in lines:

        if character_speaking:
            #print current_character
            #print line

            if comment_start:
                if ')' in line:
                    comment_start = False
                continue
            # New Comment
            elif line[:4] == '\t\t\t(':
                if ')' not in line:
                    comment_start = True
                #print 'comment:' ,[line]
                continue

            # Speaker ends
            elif line == '':
                #print 'Speaker ends;', current_character
                #print current_text
                scenes[-1].add_text(current_character,current_text)

                current_character = ''
                character_speaking = False
                current_text = ''
                #print 'Speaker ends;', [line]

            else:
                current_text += line[2:] + ' '
                #print 'else print:', [line[2:]]

        # Character speaking
        elif line[:4] == '\t\t\t\t':

            if '\t\t\t\tDISSOLVE TO:' in line:
                continue

            character_speaking = True
            current_text = ''
            current_character = line[4:]
            #print '[current_character]:', [current_character]


        # Scene ENDS
        elif line[:4] == 'EXT.' or line[:4] == 'INT.':
            scenes[-1].desc = desc
            desc = ''
            scenes.append(scene())

        # description
        else:
            desc += line.strip() + ' '
    #print lines
    #print len(scenes)

    return scenes


def parse_inception():
    script = 'inception.txt'

    raw_text = open(script,'r')
    #cleantext = BeautifulSoup(raw_text).getText()

    lines = raw_text.read().split('\n')


    scenes = []

    # add the first scene
    scenes.append(scene())
    character_speaking = False
    comment_start = False

    desc = ''

    for line in lines:

        if character_speaking:
            #print current_character
            #print line

            if comment_start:
                if ')' in line:
                    comment_start = False
                continue
            # New Comment
            elif '    (' in line:
                if ')' not in line:
                    comment_start = True
                continue

            # Speaker ends
            elif line == '':
                #print 'Speaker ends;', current_character
                #print current_text
                scenes[-1].add_text(current_character,current_text)

                current_character = ''
                character_speaking = False
                current_text = ''
                #print 'Speaker ends;', [line]

            else:
                current_text += line.strip() + ' '

        elif re.compile('[ ]+[0-9]').match(line):
            continue
        elif re.compile('[ ]+CUT TO:').match(line):
            continue
        # Character speaking
        elif line[:len('      ')] == '      ':

            #if '\t\t\t\tDISSOLVE TO:' in line:
            #    continue

            character_speaking = True
            current_text = ''
            current_character = line.strip()
            try:
                idx = current_character.index('(')
                current_character = current_character[:idx]
            except:
                pass
            current_character = current_character.strip()

        # Scene ENDS
        elif 'EXT.' == line[:4] or 'INT.' in line[:4]:

            scenes[-1].add_desc(desc)
            desc = ''
            scenes.append(scene())

        # description
        else:
            desc += line.strip() + ' '
    #print lines
    #print len(scenes)

    return scenes


def parse_shawshank():
    script = 'shawshank.txt'

    raw_text = open(script,'r')
    lines = raw_text.read().split('\n')

    scenes = []

    # add the first scene
    scenes.append(scene())
    character_speaking = False
    comment_start = False
    desc = ''

    for line in lines:
        if character_speaking:
            #print current_character
            if comment_start:
                if ')' in line:
                    comment_start = False
                continue
            # New Comment
            elif line[:4] == '\t\t\t(':
                if ')' not in line:
                    comment_start = True
                #print 'comment:' ,[line]
                continue

            # Speaker ends
            elif line == '':
                #print 'Speaker ends;', current_character
                #print current_text
                scenes[-1].add_text(current_character,current_text)

                current_character = ''
                character_speaking = False
                current_text = ''
                #print 'Speaker ends;', [line]

            else:
                current_text += line.strip() + ' '
                #print 'else print:', [line[2:]]

        # Scene ENDS
        elif re.compile('[0-9]+[\t]+(INT|EXT)').match(line):
            try:
                idx = line.index('INT') + 3
            except:
                idx = line.index('EXT') + 3
            scenes[-1].desc = desc
            desc = line[idx:].strip()
            scenes.append(scene())

        # Character speaking
        elif line[:3] == '\t\t\t':
            character_speaking = True
            current_text = ''

            current_character = line.strip()
            try:
                idx = current_character.index('(')
                current_character = current_character[:idx]
            except:
                pass
            current_character = current_character.strip()

            #print '[current_character]:', [current_character]


        # description
        else:
            desc += line.strip() + ' '
    #print lines
    #print len(scenes)

    return scenes