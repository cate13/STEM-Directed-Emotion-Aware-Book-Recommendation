from Recomender_Helper.vector_helper import get_vector_by_word, average_vectors, graphVector, cosine_similarity, get_vector_by_isbn
from create_word_level_emotion_conditioned_topic_vec import get_vocab
from Vectorizer.EmotionConditionedTopicVectorMaker import EmotionConditionedTopicVectorMaker
from get_stem_domain_books import get_isbns_with_vectors
from tqdm import tqdm
import random 
import csv

domains = ["MATHEMATICS", "PHYSICS", "BIOLOGY", "CHEMISTRY", "ENGINEERING", "COMPUTER_SCIENCE", "EARTH_SCIENCE"]
lower_domains = ["mathematics", "physics", "biology", "chemistry", "engineering", "computer_science", "earth_science"]

math_words = ["calculus", "algebra", "trigonometry", "arithmetic", "maths", "math", "basic", "algebra", "basic", "arithmetic", "mathematics", "complex", "analysis", "higher", "math", "linear", "algebra", "differential", "equations", "mathmatics", "basic", "calculus", "trig", "equations", "high", "school", "math", "imaginary", "numbers", "number", "theory", "basic", "math", "numerical", "methods", "probability", "theory", "word", "problems", "integrals", "advanced", "math", "logarithms", "formulae", "differential", "geometry", "other", "math", "real", "math", "higher", "level", "math", "theorems", "set", "theory", "group", "theory", "math", "concepts", "complex", "math", "geometry", "numerical", "analysis", "abstract", "algebra", "mathematical", "concepts", "pure", "math", "high", "school", "algebra", "measure", "theory", "basic", "logic", "Math", "topology", "matrices", "advanced", "calculus", "long", "division", "simple", "algebra", "pure", "mathematics", "arithmetic", "simple", "arithmetic", "algebraic", "topology", "number", "systems", "graph", "theory", "multivariable", "calculus", "basic", "mathematics", "actual", "math", "physics", "mathematical", "proofs", "Calculus", "infinite", "sums", "formal", "logic", "word", "problem", "combinatorics", "higher", "mathematics", "trig", "functions", "high", "school", "physics", "integral", "calculus", "advanced", "mathematics", "mathematical", "logic", "discrete", "math", "tensors", "quantum", "physics", "discrete", "mathematics", "symbolic", "logic", "fundamental", "theorem", "math", "classes", "Calculus", "mathematical", "equations", "precalculus", "enough", "math", "basic", "concepts", "Algebra", "complex", "numbers", "high", "level", "math", "partial", "differential", "equations", "quantum", "mechanics", "only", "math", "physics", "classes", "real", "analysis", "just", "math", "formulas", "algebraic", "geometry", "infinitesimals", "mathematical", "background", "quadratic", "equations", "memorization", "derivations", "factorials", "vector", "calculus", "calc", "rote", "memorization", "math", "problem", "mathematical", "reasoning", "Trigonometry", "basic", "maths", "", "Math", "mathematical", "knowledge", "intuitive", "sense", "differential", "calculus", "vector", "spaces", "field", "theory", "college", "algebra", "more", "math", "trigonometric", "functions", "classical", "mechanics", "math", "background", "Linear", "algebra", "basic", "probability", "algorithms", "basic", "chemistry", "lambda", "calculus", "Algebra", "mathematical", "polynomials", "other", "sciences", "Calculus", "matrix", "multiplication", "calculus", "class", "math", "course", "math", "stuff", "theoretical", "math", "linear", "equations", "notation", "conceptual", "understanding", "physics", "problem", "quadratics", "concepts", "mental", "math", "Euclidean", "geometry", "trig", "identities", "simple", "calculations", "basic", "statistics", "logic", "problems", "mathematical", "formulas", "partial", "derivatives", "modern", "physics", "Algebra", "QFT", "quadratic", "formula", "physics", "problems", "other", "concepts", "math", "skills", "math", "people", "important", "concepts", "precalc", "applied", "math", "Physics", "Calculus", "Real", "Analysis", "string", "theory", "pseudocode", "quadratic", "equation", "complicated", "math", "fundamental", "concepts", "functional", "analysis", "new", "math", "Maths", "text", "book", "math/physics", "Geometry", "mathematical", "formula", "Topology", "hard", "math", "mathematicians", "mental", "arithmetic", "hypothesis", "testing", "econometrics", "theoretical", "physics", "approximations", "formalism", "PDEs", "truth", "tables", "Mathematicians", "math", "major", "euclidean", "geometry", "exponents", "eigenvectors", "Geometry", "Physics", "ODEs", "math", "education", "Lagrangian", "computer", "programming", "math", "courses", "organic", "chemistry", "analytically", "Mathematics", "dynamic", "programming", "quaternions", "advanced", "physics", "calculus", "classes", "algebraically", "unit", "conversions", "modern", "mathematics", "physics", "course", "complexity", "theory", "algebraic", "pseudo-code", "statistical", "inference", "Quantum", "physics", "math", "problems", "category", "theory", "intuitive", "understanding", "Maths", "first", "principles", "mathematical", "proof", "applied", "mathematics", "physics", "class", "boolean", "logic", "mathematical", "theory", "math", "student", "pre-algebra", "university", "course", "math", "equation", "computation", "mathematical", "physics", "Calc", "2", "Quantum", "Mechanics", "introductory", "course", "simple", "understanding", "mathematical", "analysis", "logical", "statements", "dynamical", "systems", "basic", "geometry", "number", "line"]
math_words = [word.lower() for word in math_words]

physics_words = ["gravitation", "gravitational", "field", "gravitational", "fields", "black", "holes", "spacetime", "gravity", "subatomic", "particles", "gravitational", "forces", "electromagnetic", "field", "gravitational", "force", "atoms", "gravitational", "attraction", "space-time", "dark", "matter", "gravitational", "potential", "space", "time", "angular", "momentum", "elementary", "particles", "relative", "motion", "photons", "electromagnetism", "particle", "relativity", "singularity", "total", "energy", "vacuum", "energy", "gravitational", "waves", "electron", "dark", "energy", "photon", "entropy", "fundamental", "particles", "single", "particle", "gravitons", "atom", "mass/energy", "atomic", "scale", "magnetic", "field", "normal", "matter", "quantum", "level", "matter/energy", "general", "relativity", "muons", "particles", "mass-energy", "celestial", "bodies", "expanding", "universe", "gravitational", "effects", "massive", "objects", "quantum", "effects", "massive", "object", "gravitational", "effect", "atomic", "structure", "single", "electron", "quantum", "radioactive", "decay", "electrons", "inverse", "square", "law", "relativistic", "effects", "warp", "field", "singularities", "early", "universe", "physical", "matter", "energy", "states", "individual", "atoms", "massless", "antimatter", "special", "relativity", "neutrinos", "individual", "particles", "gravitational", "anti-matter", "relativistic", "speeds", "electric", "charge", "physical", "laws", "nuclear", "reactions", "tidal", "forces", "planetary", "bodies", "subatomic", "particle", "repulsive", "force", "negative", "mass", "observable", "universe", "Hawking", "radiation", "atomic", "level", "magnetic", "fields", "Black", "holes", "exotic", "matter", "solar", "system", "subatomic", "level", "warp", "bubble", "neutron", "star", "quantum", "tunneling", "gravitational", "influence", "massless", "particles", "Dark", "energy", "reference", "frame", "charged", "particles", "gravity", "waves", "energy", "conservation", "LHC", "planetary", "orbits", "rest", "frame", "single", "atom", "gravitational", "energy", "entropy", "virtual", "particles", "entanglement", "cosmic", "background", "radiation", "kinetic", "energy", "red", "shift", "quantum", "foam", "classical", "physics", "entire", "universe", "quasars", "time", "dilation", "orbits", "infinite", "mass", "universe", "cosmic", "microwave", "background", "subatomic", "galaxies", "gravitational", "pull", "quantum", "vacuum", "space/time", "quantum", "state", "inertial", "frame", "massive", "particles", "interstellar", "medium", "General", "Relativity", "neutron", "stars", "Newtonian", "physics", "star", "formation", "gravitational", "lensing", "attractive", "force", "reference", "frames", "conserved", "quantum", "field", "wavefunction", "black", "hole", "quantum", "mechanics", "hydrogen", "atom", "perturbations", "magnetism", "supernovae", "perturbation", "fundamental", "forces", "potential", "energy", "quantum", "fields", "graviton", "energy", "transfer", "conservation", "laws", "nuclei", "nuclear", "fusion", "quantum", "fluctuations", "infinite", "energy", "physical", "properties", "higgs", "field", "quantum", "particles", "fundamental", "constants", "EM", "field", "known", "universe", "perfect", "sphere", "other", "particle", "radiation", "pressure", "Photons", "phase", "space", "quarks", "finite", "speed", "macroscopic", "scale", "General", "relativity", "big", "bang", "event", "horizons", "molecules", "magnetic", "force", "electromagnetic", "forces", "light", "waves", "baryons", "electromagnetic", "waves", "distant", "stars", "relative", "velocities", "superposition", "quantum", "gravitationally", "higher", "dimensions", "curvature", "proton", "perfect", "vacuum", "fundamental", "properties", "energy", "state", "electric", "field", "energy", "output", "Newtonian", "mechanics", "electric", "and", "magnetic", "fields", "gravity", "field", "quantum", "theory", "lower", "energy", "state", "mass", "electromagnetic", "force", "other", "particles", "physical", "universe", "physical", "constants", "rest", "mass", "solar", "wind", "supermassive", "black", "holes", "Newtonian", "gravity", "escape", "velocity", "single", "photon", "EM", "radiation", "quantum", "field", "theory", "regular", "matter", "strong", "nuclear", "force", "inertial", "reference", "frames", "ordinary", "matter", "observable", "solar", "systems", "distant", "galaxies", "constant", "acceleration", "earths", "gravity", "Higgs", "field", "physical", "forces", "celestial", "objects", "physical", "law", "classical", "mechanics", "planck", "length", "quantum", "states", "visible", "universe", "uncertainty", "principle", "closed", "system", "planets", "orbiting", "galaxy", "clusters", "E=mc^2", "interferometer", "event", "horizon", "neutrino", "binding", "energy", "nuclear", "reaction", "Electrons", "earths", "wave", "function", "blackholes", "gravitational", "collapse"]
physics_words = [word.lower() for word in physics_words]

bio_words = ["organisms", "organism", "living", "organisms", "other", "organisms", "biological", "processes", "evolutionary", "history", "genes", "species", "genetic", "mutations", "humans", "living", "organism", "selection", "pressure", "genetic", "code", "genetics", "selective", "pressure", "sexual", "reproduction", "genome", "biological", "organisms", "natural", "selection", "other", "species", "genetic", "variation", "other", "organism", "evolutionary", "pressures", "natural", "processes", "genomes", "vertebrates", "cellular", "level", "evolutionary", "pressure", "selective", "pressures", "biological", "process", "random", "mutations", "genetic", "level", "single", "celled", "organisms", "microorganisms", "biological", "systems", "gene", "expression", "Organisms", "different", "organisms", "animal", "species", "mammals", "single-celled", "organisms", "genetic", "makeup", "biological", "system", "mutations", "genetic", "traits", "natural", "process", "genetic", "changes", "evolutionary", "process", "complex", "organisms", "nervous", "systems", "microbes", "physiology", "genetic", "modification", "gene", "horizontal", "gene", "transfer", "epigenetics", "human", "biology", "biological", "functions", "life", "forms", "environmental", "pressures", "living", "things", "micro-organisms", "artificial", "selection", "primates", "human", "DNA", "random", "mutation", "other", "mammals", "lifeforms", "genetic", "engineering", "microbiome", "human", "disease", "evolutionary", "prokaryotes", "bacterium", "single", "species", "genetic", "mutation", "genetic", "diversity", "human", "species", "many", "animals", "many", "species", "millions", "of", "years", "genetic", "structure", "biological", "evolution", "heredity", "&gt;Humans", "living", "cells", "sexual", "selection", "eukaryotes", "multicellular", "environmental", "factors", "selective", "breeding", "biological", "sense", "human", "genome", "specific", "genes", "Genes", "multicellular", "organisms", "particular", "species", "gametes", "certain", "species", "human", "evolution", "most", "mammals", "humans", "species", "genetic", "drift", "human", "body", "reproduction", "single", "organism", "biology", "genetic", "information", "human", "organism", "microbe", "DNA.", "fungi", "DNA.", "single", "celled", "organism", "reproductive", "cycle", "speciation", "phenotype", "different", "species", "embryonic", "development", "asexual", "reproduction", "evolution", "biological", "level", "human", "populations", "other", "animals", "evolutionary", "advantage", "living", "systems", "human", "brains", "selection", "pressures", "reproductive", "success", "", "Evolution", "evolutionary", "terms", "cell", "division", "microbiota", "most", "species", "biological", "life", "gene", "pools", "certain", "genes", "other", "genes", "hominids", "sapience", "human", "cells", "evolutionary", "processes", "human", "physiology", "gut", "bacteria", "sentience", "biosphere", "genetic", "change", "life", "form", "genetic", "manipulation", "new", "species", "phenotypes", "multiple", "species", "human", "genetics", "other", "animal", "species", "evolutionary", "sense", "own", "DNA", "human", "reproduction", "most", "animals", "sexual", "dimorphism", "chemical", "processes", "early", "humans", "invertebrates", "recombination", "human", "bodies", "Bacteria", "many", "other", "species", "other", "primates", "DNA", "sequences", "lifeform", "single", "cell", "organisms", "cognition", "homo", "sapiens", "single", "gene", "other", "living", "things", "human", "interference", "brain", "function", "Humans", "natural", "system", "only", "species", "brain", "functions", "brain", "structure", "Living", "things", "other", "apes", "natural", "world", "genetic", "programming", "metabolic", "processes", "many", "other", "animals", "Natural", "selection", "evolutionary", "trait", "most", "other", "species", "modern", "humans", "Epigenetics", "own", "genes", "human", "genes", "living", "species", "DNA", "sequence", "reproductive", "systems", "genetic", "DNA.", "plant", "cells", "photosynthesis", "organ", "systems", "human", "system", "organelles", "genotype", "physical", "processes", "mitochondria", "domestication", "neocortex", "evolutionarily", "new", "genes", "gut", "flora", "biological", "ecosystems", "evolutionary", "change", "many", "different", "species", "other", "life", "forms", "human", "brain", "mitosis", "environmental", "changes", "human", "animal", "micro", "organisms", "cognitive", "abilities", "human", "DNA.", "chemical", "reactions", "epigenetic", "mammalian", "genetically", "protozoa", "human", "development", "environmental", "influences", "biodiversity", "natural", "systems", "natural", "environment", "evolutionary", "development", "human", "society", "symbiosis", "pathogens", "plasticity", "basic", "biology", "animal", "life", "fossil", "record", "environmental", "conditions", "many", "genes", "natural", "means", "bacteria", "modern", "human", "reproductive", "system", "Human", "beings", "microorganism", "own", "genetics", "chloroplasts"]
bio_words = [word.lower() for word in bio_words]

chem_words = ["chemical", "reactions", "molecules", "atomic", "structure", "organic", "molecules", "atoms", "subatomic", "particles", "chemical", "processes", "physical", "properties", "isotopes", "chemical", "properties", "radioactive", "decay", "biological", "systems", "individual", "molecules", "electromagnetism", "organic", "compounds", "molecular", "structure", "molecule", "molecular", "level", "living", "systems", "biological", "processes", "nuclear", "reactions", "other", "molecules", "individual", "atoms", "crystal", "structure", "physical", "matter", "nuclei", "chemical", "compounds", "biological", "system", "polymers", "complex", "structures", "isotope", "chemical", "reaction", "electrons", "physical", "systems", "living", "organisms", "nuclear", "fusion", "chemical", "process", "chemical", "bonds", "nucleic", "acids", "periodic", "table", "other", "atoms", "catalysis", "chemical", "composition", "atomic", "scale", "particles", "nuclear", "fission", "complex", "molecules", "spectroscopy", "electromagnetic", "field", "dark", "matter", "fluid", "dynamics", "elementary", "particles", "biological", "organisms", "basic", "building", "blocks", "fundamental", "particles", "different", "molecules", "carbon", "atoms", "magnetic", "fields", "physical", "processes", "radioactive", "elements", "quantum", "superconductors", "nuclear", "reaction", "atom", "nanoparticles", "small", "molecules", "normal", "matter", "superconductivity", "semiconductors", "water", "molecules", "electric", "charge", "muonium", "fundamental", "properties", "deuterium", "building", "blocks", "magnetism", "particle", "accelerators", "graphene", "Brownian", "motion", "molecular", "bonds", "basic", "properties", "metals", "hydrogen", "atoms", "electronegativity", "covalent", "bonds", "crystalline", "structure", "atomic", "level", "particle", "physics", "radioactive", "isotopes", "quantum", "effects", "vacuum", "energy", "heavier", "elements", "carbon", "atom", "chemical", "interactions", "other", "particles", "material", "properties", "gravitational", "fields", "carbon", "nanotubes", "antimatter", "individual", "particles", "quantum", "particles", "physical", "process", "atomic", "nuclei", "radioactivity", "natural", "processes", "ions", "neutrons", "quantum", "phenomena", "nanotubes", "electromagnetic", "waves", "ionization", "charged", "particles", "early", "universe", "heavy", "elements", "hydrogen", "atmospheric", "oxygen", "subatomic", "level", "certain", "properties", "underlying", "physics", "physical", "phenomena", "physical", "system", "functional", "groups", "magnetic", "field", "fission", "electric", "and", "magnetic", "fields", "quantum", "field", "theory", "fusion", "reactions", "quantum", "mechanics", "Molecules", "intermolecular", "forces", "quantum", "tunneling", "anti-matter", "supernovae", "gravitational", "waves", "hydrogen", "gas", "chirality", "smaller", "particles", "black", "holes", "dark", "energy", "EM", "radiation", "physical", "structures", "other", "energy", "quarks", "exotic", "matter", "organic", "chemicals", "thermal", "energy", "oxygen", "atoms", "entropy", "hydrolysis", "photoelectric", "effect", "valence", "electrons", "electrical", "currents", "molecular", "nucleus", "other", "elements", "electrical", "charge", "field", "theory", "atomic", "theory", "hydrogen", "bonds", "photosynthesis", "electromagnetic", "fields", "living", "cells", "various", "processes", "quantum", "systems", "electron", "energy", "transfer", "statistical", "mechanics", "free", "electrons", "singularities", "mathematical", "model", "single", "molecule", "chemical", "makeup", "EM", "fields", "muons", "protons", "oxygen", "molecules", "baryons", "entanglement", "silicon", "fundamental", "forces", "free", "energy", "synthesizing", "water", "molecule", "basic", "chemistry", "energy", "input", "electromagnetic", "radiation", "microtubules", "life", "forms", "gasses", "quantum", "field", "organelles", "lighter", "elements", "gases", "gravitation", "hydrocarbons", "wavefunctions", "single", "atom", "inorganic", "hydrogen", "bonding", "regular", "matter", "electromagnetic", "forces", "carbon", "fusion", "reaction", "electric", "fields", "synthesize", "gravitons", "quantum", "physics", "hydrogen", "atom", "quantum", "states", "experiments", "certain", "metals", "mathematical", "equations", "neural", "connections", "physical", "forces", "physical", "mechanism", "superconductor", "organic", "matter", "single", "particle", "permeability", "many", "materials", "chemical", "substrates", "electric", "charges", "recombine", "electromagnetics", "macroscopic", "level", "quantum", "level", "neutrinos", "cesium", "different", "properties", "quantum", "system", "just", "atoms", "electric", "currents", "classical", "mechanics", "base", "elements", "fundamental", "physics"]
chem_words = [word.lower() for word in chem_words]

engineering_words = ["engineering", "mechanical", "engineering", "electrical", "engineering", "civil", "engineering", "computer", "science", "robotics", "chemical", "engineering", "aerospace", "industrial", "engineering", "software", "engineering", "computer", "engineering", "aerospace", "engineering", "computer", "programming", "electronic", "engineering", "biomedical", "engineering", "materials", "science", "computer", "sciences", "mechanical", "engineer", "Computer", "Science", "engineering", "informatics", "business", "management", "mechatronics", "Electrical", "Engineering", "structural", "engineering", "Electrical", "engineering", "material", "science", "Electrical", "Engineering", "applied", "physics", "related", "fields", "Computer", "science", "geophysics", "related", "field", "computer", "science", "degree", "physics", "degree", "life", "sciences", "bioengineering", "engineering", "degree", "engineering", "field", "applied", "mathematics", "business", "administration", "Computer", "Science", "comp", "sci", "Computer", "Engineering", "Chemical", "Engineering", "EE", "degree", "nuclear", "engineering", "Mechanical", "Engineering", "aerospace", "industry", "bioinformatics", "CompSci", "compsci", "Civil", "Engineering", "Mechanical", "engineering", "environmental", "engineering", "biomed", "applied", "science", "computer", "networking", "science", "degree", "astrophysics", "Software", "Engineering", "applied", "math", "science/engineering", "systems", "engineering", "math", "degree", "mechanical", "engineering", "degree", "information", "technology", "MechE", "construction", "management", "other", "fields", "ChemE", "aeronautical", "engineering", "business", "major", "aeronautics", "biomedical", "sciences", "technical", "field", "graduate", "work", "majoring", "actuarial", "science", "CS", "degree", "Biomedical", "Engineering", "earth", "science", "Mechanical", "Engineering", "engineering", "background", "computer", "engineer", "undergraduate", "theoretical", "physics", "major", "related", "courses", "Physics", "science", "field", "PhD", "other", "disciplines", "BSc", "undergraduate", "Engineering", "molecular", "biology", "many", "fields", "environmental", "science", "phd", "chemistry", "degree", "PhD", "advanced", "math", "Computer", "Engineering", "electrical", "engineering", "degree", "data", "science", "digital", "design", "Comp", "Sci", "control", "theory", "Software", "engineering", "many", "engineers", "engineering", "degrees", "programming", "Information", "Technology", "Software", "Engineering", "many", "other", "fields", "engineering", "discipline", "graduate", "degree", "petroleum", "engineering", "Engineering", "engineering", "school", "ME", "degree", "double", "major", "completely", "different", "field", "information", "systems", "product", "design", "environmental", "studies", "degree", "dual", "degree", "engineering", "physics", "business", "field", "liberal", "arts", "engineering", "design", "biophysics", "majored", "specific", "field", "art", "history", "engineering", "student", "biochemistry", "meteorology", "engineering", "disciplines", "nuclear", "physics", "IT", "degree", "finance", "mathematics", "computer", "engineering", "degree", "business", "degree", "analytical", "chemistry", "biological", "sciences", "CS", "major", "geology", "Computer", "engineering", "minoring", "biotech", "Aerospace", "Engineering", "oceanography", "electrical", "engineer", "PhD.", "CompSci", "software", "development", "senior", "design", "masters", "degree", "mechanical", "engineers", "advanced", "mathematics", "microbiology", "related", "degree", "network", "engineering", "related", "job", "hard", "sciences", "engineering", "side", "business", "school", "CS", "minor", "biomedical", "sciences", "actual", "engineering", "bachelors", "technical", "degree", "accounting", "machine", "learning", "chemical", "engineer", "specialized", "field", "Engineering", "degree", "high", "level", "math", "health", "sciences", "undergrad", "grad", "degree", "computer", "scientist", "technical", "school", "industry", "experience", "comp", "sci", "degree", "undergraduate", "degree", "Comp", "Sci", "organic", "chemistry", "supply", "chain", "management", "marine", "biology", "computer", "science", "major", "ChemE", "life", "science", "practical", "experience", "engineering", "major", "Aerospace", "Engineering", "undergraduate", "level", "biomedical", "engineer", "industrial", "design", "major", "humanities", "mathematical", "physics", "engineer", "related", "jobs", "subfield", "pure", "math", "earth", "sciences", "Mechanical", "Engineer", "different", "field", "food", "science", "various", "fields", "culinary", "arts", "PhD", "student", "math/physics", "plasma", "physics", "Environmental", "Science", "Information", "Systems", "zoology", "completely", "unrelated", "field", "other", "field", "computer", "vision", "unrelated", "field", "IT", "field", "strong", "background", "Astrophysics", "undergrad", "physics", "major", "design", "engineer", "history", "degree", "process", "engineer", "neuroscience", "senior", "project", "Biology", "software", "engineer", "Graphic", "Design", "automotive", "engineering", "engineering", "company", "physical", "science", "minored", "particular", "field", "political", "science", "graphic", "design", "research", "lab", "growing", "field"]
engineering_words = [word.lower() for word in engineering_words]

CS_words = ["software", "algorithms", "computing", "distributed", "systems", "programming", "compilers", "computer", "programs", "web", "applications", "computer", "systems", "big", "data", "databases", "software", "tools", "cryptography", "operating", "systems", "specific", "software", "existing", "software", "data", "processing", "machine", "learning", "coding", "APIs", "reverse", "engineering", "programming", "language", "programming", "languages", "open", "source", "software", "software", "embedded", "systems", "coding", "web", "apps", "implementations", "development", "environment", "new", "software", "computer", "vision", "open", "source", "code", "hardware/software", "distributed", "computing", "FPGAs", "debugging", "interfaces", "user", "interfaces", "own", "software", "natural", "language", "processing", "specialized", "software", "simple", "programs", "custom", "software", "software", "development", "computer", "hardware", "underlying", "hardware", "architect", "computers", "computer", "software", "production", "system", "neural", "nets", "code", "programing", "workflows", "automated", "testing", "GUIs", "relational", "databases", "business", "processes", "business", "process", "programming", "production", "environment", "such", "software", "computation", "software", "programs", "concurrency", "physical", "hardware", "existing", "tools", "programmer", "business", "applications", "debuggers", "software", "architecture", "commercial", "software", "deep", "learning", "computer", "security", "frameworks", "hardware", "end", "users", "TCP/IP", "programmers", "just", "software", "embedded", "devices", "debug", "network", "protocols", "other", "technologies", "image", "recognition", "web", "services", "most", "software", "game", "engines", "software", "package", "database", "design", "specific", "technology", "device", "drivers", "software", "applications", "SQL", "proprietary", "software", "legacy", "systems", "computer", "architecture", "network", "security", "interface", "toolchain", "end-users", "embedded", "software", "MATLAB", "technologies", "existing", "code", "middleware", "complex", "algorithms", "software/hardware", "cloud", "computing", "writing", "code", "API", "data", "analysis", "quantum", "computers", "business", "logic", "GUI", "hardware", "level", "web", "application", "FPGA", "complex", "systems", "TCP/IP", "scientific", "computing", "design", "patterns", "debugging", "technical", "documentation", "own", "code", "unit", "testing", "network", "protocol", "data", "storage", "system", "architecture", "GUI", "business", "requirements", "VBA", "other", "software", "actual", "software", "open", "source", "software", "project", "software", "stack", "quantum", "computing", "compiler", "best", "practices", "neural", "networks", "genetic", "algorithms", "large", "datasets", "software", "product", "CRUD", "executable", "code", "data", "structures", "architecting", "embedded", "system", "Windows", "environment", "program", "database", "management", "unit", "tests", "source", "code", "computer", "networks", "computing", "codebase", "Matlab", "software", "system", "embedded", "programming", "linux", "kernel", "computer", "code", "automated", "tests", "software", "design", "web", "technologies", "information", "security", "most", "programmers", "programing", "specific", "applications", "current", "software", "abstraction", "layer", "SQL.", "applications", "API.", "actual", "code", "certain", "software", "Machine", "learning", "MATLAB", "standard", "libraries", "open", "source", "libraries", "static", "analysis", "business", "software", "speech", "recognition", "processes", "encryption", "algorithm", "good", "software", "large", "data", "sets", "other", "people's", "code", "SDKs", "scripting", "languages", "physical", "devices", "network", "architecture", "virtual", "environment", "production", "systems", "scripting", "language", "application", "development", "SQL", "queries", "AI", "systems", "application", "code", "scalability", "basic", "coding", "other", "applications", "heuristics", "computations", "Cryptography", "ActiveRecord", "webapps", "open", "source", "tools", "most", "code", "high", "level", "languages", "Powershell", "functional", "code", "PowerShell", "machine", "code", "problem", "domain", "non-technical", "people", "software", "program", "C/C++", "architectures", "desktop", "software", "security", "vulnerabilities", "artificial", "intelligence", "hadoop", "Matlab", "system", "design", "new", "processes", "end-user", "only", "software", "Compilers", "bootstrapping", "modern", "software", "based", "applications", "interfacing", "Operating", "Systems", "custom", "hardware", "Programming", "virtualization", "SQL.", "SQL", "databases", "software", "developers", "control", "systems", "computer", "scientists", "assembly", "code", "file", "formats", "dev", "environment", "programming", "skills", "quantum", "computer", "data", "manipulation", "hardware", "device", "enterprise", "software", "open-source", "software", "SQL", "hardware", "specific", "implementation", "C", "code", "IDEs", "software", "side", "backend", "system", "administration", "softwares", "specific", "hardware", "existing", "solutions", "user", "interface", "basic", "programming", "custom", "code", "programming", "knowledge", "state", "machines", "other", "programmers"]
CS_words = [word.lower() for word in CS_words]

planet_words = ["ecology", "climate", "geology", "weather", "patterns", "water", "cycle", "human", "activity", "ocean", "currents", "oceans", "plate", "tectonics", "biodiversity", "hydrology", "soils", "carbon", "cycle", "coral", "reefs", "biosphere", "plant", "life", "ocean", "acidification", "fisheries", "volcanoes", "marine", "life", "global", "climate", "agriculture", "human", "activities", "volcanic", "activity", "rising", "temperatures", "greenhouse", "gasses", "greenhouse", "effect", "atmospheric", "composition", "greenhouse", "gases", "meteorology", "ecosystems", "climate", "changes", "weather", "systems", "ocean", "life", "oceanography", "surface", "water", "global", "climate", "change", "hydrothermal", "vents", "habitats", "environmental", "sciences", "water", "management", "volcanism", "reforestation", "sea", "life", "phytoplankton", "human", "influence", "environment", "biosphere", "natural", "cycles", "deep", "ocean", "pollution", "aquaculture", "acidification", "habitat", "global", "warming", "extreme", "weather", "geological", "geothermal", "energy", "acid", "rain", "water", "treatment", "plant", "growth", "climates", "carbon", "sequestration", "ice", "ages", "ecological", "impact", "watersheds", "pollutants", "manmade", "water", "conservation", "desertification", "terraforming", "environmental", "glaciation", "algae", "blooms", "groundwater", "deforestation", "radioactivity", "wind", "patterns", "Climate", "change", "local", "environment", "climate", "system", "ecological", "paleontology", "rainforests", "solar", "radiation", "changing", "climate", "terraforming", "industrial", "processes", "food", "production", "earth", "science", "topography", "extremophiles", "Antarctic", "environmental", "conditions", "entire", "ecosystems", "rising", "sea", "levels", "glaciers", "sustainable", "energy", "fluid", "dynamics", "aquatic", "life", "climate", "tectonic", "activity", "rain", "forests", "natural", "processes", "greenhouse", "cloud", "formation", "atmospheric", "CO2", "ocean", "water", "natural", "systems", "environmental", "impacts", "vegetation", "geoengineering", "other", "pollutants", "hydrocarbons", "polar", "ice", "caps", "water", "quality", "ground", "water", "green", "house", "gases", "materials", "science", "tropics", "natural", "environment", "currents", "rainfall", "astronomy", "atmosphere", "CO2", "levels", "reefs", "polar", "regions", "natural", "environments", "geophysics", "air", "pollution", "Global", "warming", "volcanos", "nuclear", "fusion", "geologists", "earthquakes", "volcanic", "eruptions", "earth", "sciences", "droughts", "extreme", "weather", "events", "water", "resources", "human", "impact", "wetlands", "lifeforms", "photosynthesis", "salinity", "methane", "climatology", "liquid", "water", "water", "availability", "overfishing", "fish", "populations", "extinctions", "GHGs", "soil", "erosion", "sea", "level", "rise", "sediments", "migration", "patterns", "climate", "change", "ozone", "layer", "deep", "sea", "life", "forms", "surrounding", "environment", "solar", "activity", "desalinization", "polar", "caps", "energy", "production", "mass", "extinction", "microorganisms", "space", "travel", "fossil", "fuel", "irrigation", "sedimentation", "CO2", "ice", "melting", "microbes", "forestry", "water", "pollution", "biofuels", "water", "systems", "wind", "farms", "rain", "forest", "micro-organisms", "power", "generation", "seismic", "activity", "CO2", "deep", "oceans", "energy", "generation", "petroleum", "industry", "polution", "physical", "geography", "hydroponics", "biological", "systems", "organic", "matter", "CO2", "emissions", "warming", "CO2", "tectonic", "plates", "solar", "energy", "desalination", "wind", "energy", "fracking", "seafloor", "soil", "quality", "rainforest", "material", "science", "microbial", "life", "conservation", "fossil", "fuels", "agricultural", "CO2", "water", "tables", "runaway", "greenhouse", "effect", "environmental", "concerns", "radioactive", "elements", "industrial", "agriculture", "ocean", "surface", "human", "survival", "aerosols", "ice", "age", "ecologist", "human", "health", "coastal", "areas", "tectonics", "living", "organisms", "water", "sources", "surface", "temperatures", "millions", "of", "years", "ago", "global", "environment", "environmental", "issues", "waterways", "ocean", "several", "degrees", "erosion", "arctic", "cloud", "cover", "global", "temperatures", "oil", "extraction", "greenhouse", "gas", "wind", "power", "global", "temperature", "wildfires", "animal", "agriculture", "warmer", "temperatures", "geologist", "biology", "photovoltaics", "geography", "biomass", "soil", "type", "tidal", "power"]
planet_words = [word.lower() for word in planet_words]


def parse_isbn_file(filepath='classified_isbns_limited_to_have_vec.txt'):
    categories = {}
    current_category = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if the line is a header (starts and ends with ===)
            if line.startswith('===') and line.endswith('==='):
                # Extract the category name and clean it up
                current_category = line.replace('===', '').strip()
                categories[current_category] = []
            
            # If it's an ISBN and we have an active category, add it
            elif current_category:
                categories[current_category].append(line)
                
    return categories

def get_average_word_vecs(vec_type):
    #Math
    half_size = len(math_words) // 2
    random.shuffle(math_words)
    math_sample = math_words[:half_size]
    math_test = math_words[half_size:]
    math_vecs = []
    for word in math_sample:
        math_vecs.append(get_vector_by_word(word, vec_type))

    #physics 
    half_size = len(physics_words) // 2
    random.shuffle(physics_words)
    physics_sample = physics_words[:half_size]
    physics_test = physics_words[half_size:]
    physics_vecs = []
    for word in physics_sample:
        physics_vecs.append(get_vector_by_word(word, vec_type))

    #bio
    half_size = len(bio_words) // 2
    random.shuffle(bio_words)
    bio_sample = bio_words[:half_size]
    bio_test = bio_words[half_size:]
    bio_vecs = []
    for word in bio_sample:
        bio_vecs.append(get_vector_by_word(word, vec_type))
    
    #chem
    half_size = len(chem_words) // 2
    random.shuffle(chem_words)
    chem_sample = chem_words[:half_size]
    chem_test = chem_words[half_size:]
    chem_vecs = []
    for word in chem_sample:
        chem_vecs.append(get_vector_by_word(word, vec_type))

    #engineering
    half_size = len(engineering_words) // 2
    random.shuffle(engineering_words)
    engineering_sample = engineering_words[:half_size]
    engineering_test = engineering_words[half_size:]
    engineering_vecs = []
    for word in engineering_sample:
        engineering_vecs.append(get_vector_by_word(word, vec_type))

    #CS
    half_size = len(CS_words) // 2
    random.shuffle(CS_words)
    CS_sample = CS_words[:half_size]
    CS_test = CS_words[half_size:]
    CS_vecs = []
    for word in CS_sample:
        CS_vecs.append(get_vector_by_word(word, vec_type))

    #planet
    half_size = len(planet_words) // 2
    random.shuffle(planet_words)
    planet_sample = planet_words[:half_size]
    planet_test = planet_words[half_size:]
    planet_vecs = []
    for word in planet_sample:
        planet_vecs.append(get_vector_by_word(word, vec_type))

    return [average_vectors(math_vecs), average_vectors(physics_vecs), average_vectors(bio_vecs), average_vectors(chem_vecs), average_vectors(engineering_vecs), average_vectors(CS_vecs), average_vectors(planet_vecs)], [math_test, physics_test, bio_test, chem_test, engineering_test, CS_test, planet_test], ["MATH", "PHYSICS", "BIO", "CHEM", "ENGINEERING", "CS", "PLANET"]



def get_average_vec(domain, data):
    isbns = data[domain]
    mid_point = len(isbns) // 2
    random.shuffle(isbns)
    isbns_sample = isbns[:mid_point]
    isbns_test = isbns[mid_point:]
    # emotion_conditioned_vector_maker = EmotionConditionedTopicVectorMaker()

    vec_to_average = []
    for i in isbns_sample:
        # e_vec = get_vector_by_isbn(i, "emotion")
        t_vec = get_vector_by_isbn(i, "tf_idf")
        #vec_to_average.append(emotion_conditioned_vector_maker.get_emotion_conditioned_topic_vector(t_vec, e_vec))
        vec_to_average.append(t_vec)

    return average_vectors(vec_to_average), isbns_test

def book_domain_accuracy_by_domain():
    data = parse_isbn_file()
    key_view = data.keys()
    print(f"Keys as view object: {key_view}")
    
    average_vec_list = []
    list_remaining_lists = []

    # lower_domains is used here to build our reference vectors
    for d in lower_domains:
        av_vec, remaining_isbns = get_average_vec(d, data)
        average_vec_list.append(av_vec)
        list_remaining_lists.append(remaining_isbns)
    
    total_books = 0
    correct_predictions = 0
    
    # Dictionary to track stats per domain
    domain_results = {}

    for i in range(len(list_remaining_lists)):
        current_books = list_remaining_lists[i]
        current_domain = lower_domains[i]
        
        # Initialize the tracker for this specific domain
        domain_results[current_domain] = {"correct": 0, "total": 0}

        for book in tqdm(current_books, desc=f"Classifying {current_domain}"):
            book_t_vec = get_vector_by_isbn(book, "tf_idf")
            vec = book_t_vec

            total_books += 1
            domain_results[current_domain]["total"] += 1

            best_similarity = -1.0
            predicted_index = -1

            for j, avg_vec in enumerate(average_vec_list):
                similarity = cosine_similarity(vec, avg_vec)

                if similarity > best_similarity:
                    best_similarity = similarity
                    predicted_index = j
            
            if predicted_index == i:
                correct_predictions += 1
                domain_results[current_domain]["correct"] += 1

    # Calculate final accuracy percentages for each domain
    for domain, stats in domain_results.items():
        if stats["total"] > 0:
            stats["accuracy"] = (stats["correct"] / stats["total"]) * 100
        else:
            stats["accuracy"] = 0

    overall_accuracy = (correct_predictions / total_books) * 100 if total_books > 0 else 0
    
    return overall_accuracy, domain_results

def book_domain_classifier():
    data = parse_isbn_file()
    key_view = data.keys()
    print(f"Keys as view object: {key_view}")
    
    average_vec_list = []
    list_remaining_lists = []

    for d in lower_domains:
        av_vec, remaining_isbns = get_average_vec(d, data)
        average_vec_list.append(av_vec)
        list_remaining_lists.append(remaining_isbns)
    
    total_books = 0
    correct_predictions = 0
    #emotion_conditioned_vector_maker = EmotionConditionedTopicVectorMaker()

    for i in range(len(list_remaining_lists)):
        current_books = list_remaining_lists[i]

        for book in tqdm(current_books):
            #book_e_vec = get_vector_by_isbn(book, "emotion")
            book_t_vec = get_vector_by_isbn(book, "tf_idf")

            #vec = emotion_conditioned_vector_maker.get_emotion_conditioned_topic_vector(book_t_vec, book_e_vec)
            vec = book_t_vec

            total_books += 1

            best_similarity = -1.0
            predicted_index = -1

            for j, avg_vec in enumerate(average_vec_list):
                similarity = cosine_similarity(vec, avg_vec)

                if similarity > best_similarity:
                    best_similarity = similarity
                    predicted_index = j
            
            # 3. Check if the closest average vector matches the original group index
            if predicted_index == i:
                correct_predictions += 1

    accuracy = (correct_predictions / total_books) * 100 if total_books > 0 else 0
    return accuracy, correct_predictions, total_books

def book_stem_classifier():
    data = parse_isbn_file()
    average_vec_list = []
    list_remaining_lists = []

    for d in lower_domains:
        av_vec, remaining_isbns = get_average_vec(d, data)
        average_vec_list.append(av_vec)
        list_remaining_lists.append(remaining_isbns)
    stem_target_vec = average_vectors(average_vec_list)
    stem_books = [item for sublist in list_remaining_lists for item in sublist]
    
    temp = get_isbns_with_vectors()
    stem_temp = set(stem_books)
    no_stem_set = temp - stem_temp
    temp_list = list(no_stem_set)
    random.shuffle(temp_list)
    non_stem = temp_list[:len(stem_books)]

    stem_scores = []
    for book in stem_books:
        vec = get_vector_by_isbn(book, "tf_idf")
        score = cosine_similarity(vec, stem_target_vec)
        stem_scores.append(score)

    target_recall = 0.90
    stem_scores.sort()
    threshold_index = int(len(stem_scores) * (1.0 - target_recall))
    cutoff = stem_scores[threshold_index]

    non_stem_matches = 0
    for book in non_stem:
        vec = get_vector_by_isbn(book, "tf_idf")
        score = cosine_similarity(vec, stem_target_vec)
        if score >= cutoff:
            non_stem_matches += 1
    
    print(f"Required Cutoff for {target_recall} Recall: {cutoff:.4f}")
    print(f"Non-STEM words flagged as STEM (False Positives): {non_stem_matches}")
    print(f"False Positive Rate: {(non_stem_matches / len(non_stem)) * 100:.2f}%")
    



def see_word_cossine_sim_test(vec_type = 'w2v_emotion_conditioned_topic_vec'):
    average_vec_list, list_remaining_lists, label_list = get_average_word_vecs(vec_type)

    with open('test_words_stem_domain.txt', 'w', encoding='utf-8') as f:
        for target_vec, test_list, label in zip(average_vec_list, list_remaining_lists, label_list):
            for word in test_list:
                word_vec = get_vector_by_word(word, vec_type)
                try:
                    cos_sim = cosine_similarity(target_vec, word_vec)
                except Exception:
                    print(f"Word: {word}")
                f.write(f"Domain {label} {word}: {cos_sim}" + '\n')

def see_word_cossine_sim_vocab(vec_type = 'w2v_emotion_conditioned_topic_vec'):
    average_vec_list, list_remaining_lists, label_list = get_average_word_vecs(vec_type)
    vocab = get_vocab()

    header = ["word", "MATH", "PHYSICS", "BIO", "CHEM", "ENGINEERING", "CS", "PLANET"]

    with open('processed_data/vocab_emotion_conditioned_topic_vec.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for word in tqdm(vocab):
            if len(word) == 0: continue
            line = []
            line.append(word)
            temp = get_vector_by_word(word, vec_type)
            for average_vec in average_vec_list:
                # print(f"Temp: {type(temp)}, {len(temp)}")
                # print(f"Temp: {type(average_vec)}, {len(average_vec)}")
                line.append(str(cosine_similarity(temp, average_vec)))
            csv_writer.writerow(line)

def word_stem_classifier(vec_type = 'w2v_topic_vec'):
    average_vec_list, list_remaining_lists, label_list = get_average_word_vecs(vec_type)
    stem_target_vec = average_vectors(average_vec_list)
    stem_words = [item for sublist in list_remaining_lists for item in sublist]
    temp = list(get_vocab())
    random.shuffle(temp)
    non_stem_words = temp[:len(stem_words)]
    
    stem_scores = []
    for word in stem_words:
        if len(word) == 0: continue 
        vec = get_vector_by_word(word, vec_type)
        score = cosine_similarity(vec, stem_target_vec)
        stem_scores.append(score)

    target_recall = 0.70
    stem_scores.sort()
    threshold_index = int(len(stem_scores) * (1.0 - target_recall))
    cutoff = stem_scores[threshold_index]

    non_stem_matches = 0
    for word in non_stem_words:
        if len(word) == 0: continue 
        vec = get_vector_by_word(word, vec_type)
        score = cosine_similarity(vec, stem_target_vec)
        if score >= cutoff:
            non_stem_matches += 1
    
    print(f"Required Cutoff for {target_recall} Recall: {cutoff:.4f}")
    print(f"Non-STEM words flagged as STEM (False Positives): {non_stem_matches}")
    print(f"False Positive Rate: {(non_stem_matches / len(non_stem_words)) * 100:.2f}%")


def word_domain_classifier(vec_type = 'w2v_topic_vec'):
    average_vec_list, list_remaining_lists, label_list = get_average_word_vecs(vec_type)

    total_words = 0
    correct_predictions = 0

    for i in range(len(list_remaining_lists)):
        current_words = list_remaining_lists[i]

        for word in tqdm(current_words):
            if len(word) == 0: continue
            word_vec = get_vector_by_word(word, vec_type)

            total_words += 1
            
            best_similarity = -1.0
            predicted_index = -1
            
            # 2. Compare word_vec against ALL average vectors
            for j, avg_vec in enumerate(average_vec_list):
                similarity = cosine_similarity(word_vec, avg_vec)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    predicted_index = j
            
            # 3. Check if the closest average vector matches the original group index
            if predicted_index == i:
                correct_predictions += 1

    accuracy = (correct_predictions / total_words) * 100 if total_words > 0 else 0
    return accuracy, correct_predictions, total_words


def word_domain_classifier_check_by_domain(vec_type = 'w2v_topic_vec'):
    average_vec_list, list_remaining_lists, label_list = get_average_word_vecs(vec_type)

    total_words = 0
    correct_predictions = 0
    
    # Dictionary to store results per domain
    # Structure: { domain_name: {"correct": 0, "total": 0} }
    domain_stats = {}

    for i in range(len(list_remaining_lists)):
        current_words = list_remaining_lists[i]
        current_domain = label_list[i]
        
        # Initialize stats for this specific domain
        domain_stats[current_domain] = {"correct": 0, "total": 0}

        for word in tqdm(current_words, desc=f"Processing {current_domain}"):
            if len(word) == 0: continue
            word_vec = get_vector_by_word(word, vec_type)

            total_words += 1
            domain_stats[current_domain]["total"] += 1
            
            best_similarity = -1.0
            predicted_index = -1
            
            for j, avg_vec in enumerate(average_vec_list):
                similarity = cosine_similarity(word_vec, avg_vec)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    predicted_index = j
            
            if predicted_index == i:
                correct_predictions += 1
                domain_stats[current_domain]["correct"] += 1

    # Calculate final accuracy for each domain
    for domain, stats in domain_stats.items():
        corr = stats["correct"]
        tot = stats["total"]
        stats["accuracy"] = (corr / tot * 100) if tot > 0 else 0

    overall_accuracy = (correct_predictions / total_words) * 100 if total_words > 0 else 0
    
    return overall_accuracy, domain_stats

# accuracy, correct, total = book_domain_classifier()
# print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

# acc, stats = book_domain_accuracy_by_domain()
# print(f"Overall Accuracy: {acc:.2f}%")

# for domain, data in stats.items():
#     print(f"Domain: {domain} | Accuracy: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")

book_stem_classifier()