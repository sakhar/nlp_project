class scene:

    def __init__(self):
        self.characters = []
        self.texts = []
        self.desc = ''
        self.persons = []
        self.sentiments = []
        self.pairs = []

    def add_text(self,character, text):
        if character not in self.characters:
            self.characters.append(character)
        self.texts.append((character,text))

    def add_desc(self,desc):
        self.desc = desc

    def add_person(self,person):
        if person.upper() not in self.persons:
            self.persons.append(person.upper())

    def add_sentiment(self, sentiment):
        self.sentiments.append(sentiment)

    def add_pair(self,pair):
        self.pairs.append(pair)