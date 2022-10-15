# nba_perf_predictor_tfm
A project to create a Machine Learning model (Random Forest) to predict the performance of NBA players during their fisrt 4 years at the NBA (rookie contract). To do so, we train the model on the performance of the players at the university leagues.

# How to run the project
To run the project the only script required to run is main.py

# What are the repository packages requirements?
Install the packages displayed in the requirements.txt file running the command:
    pip install -r requirements.txt

# What is the config.yaml file?
The config.yaml file is the file which dictates how the project will run.

It is a dictionary determining which pipelines and steps will run.

It also contaisn all the main parameters of the project such as which draft class should be predicted or how many folds are studied in the Cross validation.

# Which are the project pipelines?
There are 3 pipelines in this project, each one defined by a python class:

    - Web Scraping Pipeline: defined by the BasketScraper class in src/components/nba_scraper/BasketScraper.py
    - Featureset Generation: defined by the BasketDataProcessor class in src/components/data_processing/BasketballDataProcessor.py
    - Modeling Pipeline: defined by the DraftForecast class in src/components/modeling/DraftForecast.py
