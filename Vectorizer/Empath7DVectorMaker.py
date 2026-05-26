from empath import Empath

class Empath7DVectorMaker:
    def __init__(self):
        self.lexicon = Empath()

        self.lexicon.create_category("mathematics", ["math", "numbers", "patterns", "puzzles", "logic", "arithmetic", "algebra", "geometry", "calculus", "statistics", "trigonometry", "problem", "tessellated", "quantitative_problem_solving"], model="reddit", size=300)

        self.lexicon.create_category("physics", ["physics", "astronomy", "physical_laws", "gravity", "motion", "energy", "space_exploration", "planets", "solar_system", "entropy", "voltage", "photon", "atom", "inertia", "density", "quantum"], model="reddit", size=300)

        self.lexicon.create_category("biology", ["biology", "organisms", "genetics", "ecosystems", "evolution", "animals", "plants", "biological_discovery", "DNA", "homeostasis", "chromosome", "physiology"], model="reddit", size=300)        

        self.lexicon.create_category("chemistry", ["chemistry", "materials_science", "chemical_reactions", "molecules", "atoms", "materials", "laboratory_experiments", "element", "catalyst", "periodic_table"], model="reddit", size=300)

        self.lexicon.create_category("engineering", ["engineering", "chemical_engineering", "electrical_engineering", "mechanical_engineering", "civil_engineering", "engineering_design", "machines", "robotics", "invention"], model="reddit", size=300)

        self.lexicon.create_category("computer_science", ["computer_science", "programming", "algorithms", "computer_systems", "artificial_intelligence", "data_science", "coding", "digital", "technology", "hardware", "software", "database", "debug", "encryption"], model="reddit", size=300)

        self.lexicon.create_category("earth_science", ["earth_science", "environmental", "climate", "sustainability", "geology", "oceans", "atmosphere", "volcano", "humidity", "ecology"], model="reddit", size=300)


    def getEmapthVector(self, text, categories=["mathematics", "physics", "biology", "chemistry", "engineering", "computer_science", "earth_science"]):
        return self.lexicon.analyze(text, categories=categories, normalize=True)

temp = Empath7DVectorMaker()

# print(temp.getEmapthVector("Testing and work and math and school and learning and such"))