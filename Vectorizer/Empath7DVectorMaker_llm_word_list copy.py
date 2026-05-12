from empath import Empath

class Empath7D_GPT_Wordlist_VectorMaker:
    def __init__(self):
        self.lexicon = Empath()

        self.lexicon.create_category("mathematics", [
                                                    "algebra",
                                                    "geometry",
                                                    "equation",
                                                    "fraction",
                                                    "number",
                                                    "probability",
                                                    "pattern",
                                                    "logic",
                                                    "formula",
                                                    "puzzle",
                                                    "statistic",
                                                    "theorem",
                                                    "calculation",
                                                    "graph",
                                                    "infinity",
                                                    ], model="reddit", size=300)

        self.lexicon.create_category("physics", [
                                                "gravity",
                                                "energy",
                                                "motion",
                                                "force",
                                                "velocity",
                                                "quantum",
                                                "particle",
                                                "atom",
                                                "orbit",
                                                "relativity",
                                                "magnetism",
                                                "friction",
                                                "wave",
                                                "matter",
                                                "radiation",
                                                ], model="reddit", size=300)

        self.lexicon.create_category("biology", [
                                                "cell",
                                                "dna",
                                                "gene",
                                                "evolution",
                                                "species",
                                                "ecosystem",
                                                "organism",
                                                "anatomy",
                                                "mutation",
                                                "microscope",
                                                "habitat",
                                                "virus",
                                                "bacteria",
                                                "heredity",
                                                "adaptation",
                                                 ], model="reddit", size=300)        

        self.lexicon.create_category("chemistry", [
                                                    "atom",
                                                    "molecule",
                                                    "reaction",
                                                    "element",
                                                    "compound",
                                                    "acid",
                                                    "solution",
                                                    "laboratory",
                                                    "chemical",
                                                    "formula",
                                                    "catalyst",
                                                    "experiment",
                                                    "periodic",
                                                    "bond",
                                                    "matter",
                                                ], model="reddit", size=300)

        self.lexicon.create_category("engineering", [
                                                    "design",
                                                    "machine",
                                                    "structure",
                                                    "prototype",
                                                    "inventor",
                                                    "robot",
                                                    "circuit",
                                                    "blueprint",
                                                    "engine",
                                                    "construction",
                                                    "mechanic",
                                                    "technology",
                                                    "bridge",
                                                    "system",
                                                    "innovation",
                                                    ], model="reddit", size=300)

        self.lexicon.create_category("computer_science", [
                                                        "code",
                                                        "programming",
                                                        "algorithm",
                                                        "software",
                                                        "computer",
                                                        "data",
                                                        "network",
                                                        "debugging",
                                                        "application",
                                                        "system",
                                                        "database",
                                                        "cybersecurity",
                                                        "interface",
                                                        "artificial intelligence",
                                                        "binary",
                                                        ], model="reddit", size=300)

        self.lexicon.create_category("earth_science", [
                                                        "geology",
                                                        "rock",
                                                        "mineral",
                                                        "earthquake",
                                                        "volcano",
                                                        "plate",
                                                        "tectonics",
                                                        "fossil",
                                                        "erosion",
                                                        "atmosphere",
                                                        "climate",
                                                        "weather",
                                                        "ocean",
                                                        "soil",
                                                        "glacier",
                                                       ], model="reddit", size=300)


    def getEmapthVector(self, text, categories=["mathematics", "physics", "biology", "chemistry", "engineering", "computer_science", "earth_science"]):
        return self.lexicon.analyze(text, categories=categories, normalize=True)

#temp = Empath7DVectorMaker()

# print(temp.getEmapthVector("Testing and work and math and school and learning and such"))