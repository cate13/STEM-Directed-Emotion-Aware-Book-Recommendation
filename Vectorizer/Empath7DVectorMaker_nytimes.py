from empath import Empath

class Empath7D_NWTimes_VectorMaker:
    def __init__(self):
        self.lexicon = Empath()

        self.lexicon.create_category("mathematics", ["math", "mathematics"], model="nytimes", size=300)

        self.lexicon.create_category("physics", ["physics"], model="nytimes", size=300)

        self.lexicon.create_category("biology", ["biology"], model="nytimes", size=300)        

        self.lexicon.create_category("chemistry", ["chemistry"], model="nytimes", size=300)

        self.lexicon.create_category("engineering", ["engineering"], model="nytimes", size=300)

        self.lexicon.create_category("computer_science", ["computer science"], model="nytimes", size=300)

        self.lexicon.create_category("earth_science", ["earth science"], model="nytimes", size=300)

    def getEmapthVector(self, text, categories=["mathematics", "physics", "biology", "chemistry", "engineering", "computer_science", "earth_science"]):
        return self.lexicon.analyze(text, categories=categories, normalize=True)

#temp = Empath7DVectorMaker()

# print(temp.getEmapthVector("Testing and work and math and school and learning and such"))