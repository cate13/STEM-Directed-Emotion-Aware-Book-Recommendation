from empath import Empath

class Empath7D_LLM_Wordlist_VectorMaker:
    def __init__(self):
        self.lexicon = Empath()

        self.lexicon.create_category("mathematics", ["mathematics", "algebra", "equation", "probability", "logic", "theorem"], model="reddit", size=300)

        self.lexicon.create_category("physics", ["physics", "gravity", "quantum", "particle", "relativity", "magnetism", "friction"], model="reddit", size=300)

        self.lexicon.create_category("biology", ["biology", "cell", "DNA", "evolution", "species", "ecosystem", "organism", "anatomy", "mutation", "habitat"], model="reddit", size=300)        

        self.lexicon.create_category("chemistry", ["chemistry", "molecule", "reaction", "element", "compound", "catalyst"], model="reddit", size=300)

        self.lexicon.create_category("engineering", ["engineering", "structure", "blueprint", "innovation"], model="reddit", size=300)

        self.lexicon.create_category("computer_science", ["computer science", "network", "database", "interface"], model="reddit", size=300)

        self.lexicon.create_category("earth_science", ["earth science", "mineral", "fossil", "erosion", "atmosphere", "climate", "glacier"], model="reddit", size=300)


    def getEmapthVector(self, text, categories=["mathematics", "physics", "biology", "chemistry", "engineering", "computer_science", "earth_science"]):
        return self.lexicon.analyze(text, categories=categories, normalize=True)

#temp = Empath7DVectorMaker()

# print(temp.getEmapthVector("Testing and work and math and school and learning and such"))