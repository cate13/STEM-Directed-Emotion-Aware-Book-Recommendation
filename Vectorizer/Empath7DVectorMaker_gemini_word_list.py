from empath import Empath

class Empath7D_Gemini_Wordlist_VectorMaker:
    def __init__(self):
        self.lexicon = Empath()

        self.lexicon.create_category("mathematics", [
                                                    "algorithm",
                                                    "probability",
                                                    "fractal",
                                                    "theorem",
                                                    "variable",
                                                    "symmetry",
                                                    "coordinate",
                                                    "dimension",
                                                    "infinite",
                                                    "sequence",
                                                    "equation",
                                                    "cryptography",
                                                    "logic",
                                                    "optimization",
                                                    "topology",
                                                    ], model="reddit", size=300)

        self.lexicon.create_category("physics", [
                                                "velocity",
                                                "momentum",
                                                "gravity",
                                                "entropy",
                                                "singularity",
                                                "quantum",
                                                "kinetic",
                                                "friction",
                                                "relativity",
                                                "magnetism",
                                                "frequency",
                                                "particle",
                                                "mass",
                                                "inertia",
                                                "trajectory",
                                                ], model="reddit", size=300)

        self.lexicon.create_category("biology", [
                                                "evolution",
                                                "genetic",
                                                "organism",
                                                "mutation",
                                                "ecosystem",
                                                "anatomy",
                                                "cell",
                                                "diversity",
                                                "microbe",
                                                "habitat",
                                                "species",
                                                "metabolism",
                                                "parasite",
                                                "symbiosis",
                                                "dna",
                                                 ], model="reddit", size=300)        

        self.lexicon.create_category("chemistry", [
                                                    "element",
                                                    "molecule",
                                                    "reaction",
                                                    "catalyst",
                                                    "solubility",
                                                    "acidic",
                                                    "bond",
                                                    "isotope",
                                                    "compound",
                                                    "synthetic",
                                                    "aqueous",
                                                    "valence",
                                                    "mixture",
                                                    "distill",
                                                    "polymer",
                                                ], model="reddit", size=300)

        self.lexicon.create_category("engineering", [
                                                    "prototype",
                                                    "infrastructure",
                                                    "mechanics",
                                                    "schematic",
                                                    "efficiency",
                                                    "structure",
                                                    "blueprint",
                                                    "automation",
                                                    "aerospace",
                                                    "robotics",
                                                    "component",
                                                    "thermal",
                                                    "hydraulics",
                                                    "innovation",
                                                    "systems",
                                                    ], model="reddit", size=300)

        self.lexicon.create_category("computer_science", [
                                                        "hardware",
                                                        "database",
                                                        "encryption",
                                                        "interface",
                                                        "compiler",
                                                        "network",
                                                        "firmware",
                                                        "backend",
                                                        "logic",
                                                        "script",
                                                        "debug",
                                                        "recursion",
                                                        "protocol",
                                                        "variable",
                                                        "syntax",
                                                        ], model="reddit", size=300)

        self.lexicon.create_category("earth_science", [
                                                        "tectonic",
                                                        "magma",
                                                        "erosion",
                                                        "sedimentary",
                                                        "atmosphere",
                                                        "biosphere",
                                                        "lithosphere",
                                                        "glacier",
                                                        "seismic",
                                                        "fossil",
                                                        "crust",
                                                        "topography",
                                                        "climate",
                                                        "geothermal",
                                                        "mineral",
                                                       ], model="reddit", size=300)


    def getEmapthVector(self, text, categories=["mathematics", "physics", "biology", "chemistry", "engineering", "computer_science", "earth_science"]):
        return self.lexicon.analyze(text, categories=categories, normalize=True)

#temp = Empath7DVectorMaker()

# print(temp.getEmapthVector("Testing and work and math and school and learning and such"))