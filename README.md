# Emotion-Aware and Topic-Directed Book Recommender System

This repository contains the implementation, datasets, evaluation code, and supporting analyses for an emotion-aware and topic-directed book recommendation system. The project focuses on recommending books to teen readers, with an emphasis on identifying and prioritizing STEM-related books.

## Repository Structure

### `Classifier/`

Contains code and datasets for training machine learning classifiers to identify books as **STEM** or **Not STEM**.

This folder includes:

- Classifier training and evaluation code
- Datasets used to train the classifiers
- Supporting data for STEM classification experiments

### `data_exploring/`

Contains scripts used to explore and analyze the datasets.

Examples of data exploration tasks include:

- Identifying the age ranges of users in the dataset
- Finding and analyzing book topics
- Identifying users who have read specific sets of books
- Exploring user and book characteristics
- Performing other exploratory data analysis

### `Implementation/`

Contains the primary implementation of the book recommendation system and its evaluation.

This folder includes:

- Code for generating book vectors
- Book recommendation algorithms
- Recommendation evaluation and metric calculation
- Supporting implementation code

The primary recommendation code is located at `Implementation/recommender.py`.

The evaluation code is located at `Implementation/metrics.py`.

### `LLM Code/`

Contains four Python programs for generating book recommendations using large language models (LLMs).

Each program uses a different set of instructions:

1. **No special instructions**
   - Generates recommendations without explicitly prioritizing STEM books.

2. **STEM prioritization**
   - Instructs the LLM to prioritize STEM books.

3. **STEM prioritization using STEM book vectors**
   - Instructs the LLM to prioritize STEM books using STEM-related book vector information.

4. **STEM and emotion vector prioritization**
   - Instructs the LLM to prioritize STEM books using both STEM and emotion-related book vector information.

These experiments allow different approaches to incorporating STEM relevance and emotional preferences into LLM-based recommendations to be compared.

### `processed_data/`

Contains processed datasets and intermediate data generated throughout the project.

This folder includes:

- JSONL files containing generated book vectors
- Code for creating user lists
- JSONL files containing user lists
- TXT files containing ISBNs identified as STEM

### `Recommender_Helper/`

Contains helper functions used throughout the recommendation system.

The primary file is `Recommender_Helper/vector_helper.py`.

This file contains various utility methods for creating, loading, retrieving, and processing book vectors.

### `recommendations/`

Contains generated book recommendation results from the various recommendation experiments.

These files are used as inputs to the recommendation evaluation code.

### `STEM_Annontate/`

Contains the materials and results from the manual annotation of books as STEM or Not STEM.

This folder includes:

- Manual annotation instructions
- CSV files containing annotated books
- Code for evaluating inter-rater agreement using Fleiss' kappa

### `STEM_Books/`

Contains code and data used to identify STEM books based on subject, tag, and genre information.

This folder includes:

- Code for retrieving tags and genre information from the Library of Congress
- Code for retrieving genre information from Google Books
- Code for categorizing tags and genre information as STEM or Not STEM
- Comprehensive lists of tags and genres
- Lists of tags and genres identified as STEM

### `stem_emotion_relation/`

Contains code for exploring the relationship between users' experiences with STEM books and their emotional preferences.

This analysis investigates how emotional preferences may relate to users' interactions with STEM-related books.

### `user_eval_sets/`

Contains JSON files defining the user test sets used to evaluate the recommendation system.

These files contain the book interactions used for evaluating recommendation performance.

### `Vectorizer/`

Contains classes used to convert book descriptions and other book information into different types of book vector representations.

These vector representations are used as inputs to the recommendation system.

### `environment.yml`

Conda environment configuration file containing the dependencies required to run the project.

Create the environment using:

`conda env create -f environment.yml`

Then activate the environment using:

`conda activate <environment-name>`

---

## Generating Book Recommendations

The primary recommendation implementation is located at `Implementation/recommender.py`.

This file contains several different recommendation methods for generating book recommendations using different combinations of book vectors and recommendation strategies.

### Selecting the Test Set

The test set used during recommendation generation is specified using the `TEST_DATA_FILE` variable.

Change this variable to select the user evaluation dataset you want to use.

### Running Recommendation Experiments

The `run_base_combo()` method runs four different test configurations.

These configurations allow the recommendation system to be evaluated using different combinations of the available book vector representations and recommendation approaches.

Many book recommendation experiments have already been completed. Their results are stored in the `recommendations/` directory.

Therefore, it is generally not necessary to regenerate existing recommendation results unless you want to reproduce or modify an experiment.

---

## Evaluating Book Recommendations

Recommendation evaluation is implemented in `Implementation/metrics.py`.

This file contains methods for calculating the performance metrics used to evaluate recommendation results.

Two particularly important methods are `aggregate_metrics_to_csv()` and `compare_baseline_to_others()`.

### `aggregate_metrics_to_csv()`

This method evaluates all recommendation results contained in a specified folder and aggregates their metrics into a CSV file.

This is useful for comparing the overall performance of multiple recommendation configurations.

The evaluation includes metrics such as:

- Mean Reciprocal Rank (MRR)
- Precision at 1 (P@1)
- Precision at 3 (P@3)
- Precision at 5 (P@5)
- Spearman correlation
- NDCG at different cutoff values

### `compare_baseline_to_others()`

This method compares a specific recommendation result against the other recommendation results in a specified folder.

This can be used to evaluate how significant the results are compared to baseline. 

---

## Project Workflow

The general workflow of the project is:

**Raw Book/User Data**  
↓  
**Data Exploration**  
↓  
**STEM Book Identification and Classification**  
↓  
**Book Vector Generation**  
↓  
**User Evaluation Set Creation**  
↓  
**Book Recommendation**  
↓  
**Recommendation Results**  
↓  
**Metric Evaluation**  
↓  
**Comparison of Recommendation Methods**

The project also includes a separate LLM-based recommendation workflow, which uses different prompting strategies and combinations of STEM and emotion-related book information.

---

## Reproducibility

To reproduce the experiments:

1. Create the required Conda environment using `environment.yml`.
2. Prepare or obtain the required source datasets which comes from the [Book Crossing dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset) 
3. Generate the required book vectors using the code in `Implementation/` and `Vectorizer/`. (Many of the book vectors are already generated and found in `procssed_data/`)
4. Prepare the user evaluation sets in `user_eval_sets/`. (Many user evaluations sets are already generated and found in `procssed_data/`)
5. Configure the desired test set using the `TEST_DATA_FILE` variable in `Implementation/recommender.py`.
6. Run the desired recommendation method.
7. Save the generated recommendations in the `recommendations/` directory.
8. Use the evaluation methods in `Implementation/metrics.py` to calculate and compare recommendation performance.




