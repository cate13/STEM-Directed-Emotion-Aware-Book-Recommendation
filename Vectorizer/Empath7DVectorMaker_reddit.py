from empath import Empath

class Empath7DRedditVectorMaker:
    def __init__(self):
        self.lexicon = Empath()

        self.lexicon.create_category("mathematics", model="reddit", size=300)

        self.lexicon.create_category("physics", model="reddit", size=300)

        self.lexicon.create_category("biology", model="reddit", size=300)        

        self.lexicon.create_category("chemistry", model="reddit", size=300)

        self.lexicon.create_category("engineering", model="reddit", size=300)

        self.lexicon.create_category("computer_science", model="reddit", size=300)

        self.lexicon.create_category("earth_science", model="reddit", size=300)


    def getEmapthVector(self, text, categories=["mathematics", "physics", "biology", "chemistry", "engineering", "computer_science", "earth_science"]):
        return self.lexicon.analyze(text, categories=categories, normalize=True)

#temp = Empath7DVectorMaker()

# print(temp.getEmapthVector("Testing and work and math and school and learning and such"))