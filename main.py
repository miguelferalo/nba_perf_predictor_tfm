from config import config_variables
from src.components.nba_scrapper.BasketScraper import BasketScrapper
from src.components.data_preprocessing.BasketballDataProcessor import BasketDataProcessor

######################## SET A WORKFLOW IN THE PROJECT #################################

############ PIPELINE TO SCRAPE DATA ############
if config_variables['FLOW_SWITCH']['WEB_SCRAPPING']:

    basketball_scrapper = BasketScrapper(config_variables=config_variables)

    if config_variables['SCRAPING_VARS']['CONTROL']['COLLEGE_SCRAPPING']:

        #Get Data of college players
        basketball_scrapper.college_data_scraping()

    if config_variables['SCRAPING_VARS']['CONTROL']['NBA_SCRAPING']:
        #Get Data of NBA players
        basketball_scrapper.nba_data_scraping()

############ PIPELINE TO CREATE FEATURESET, IMPUTE, VISUALIZE CORRELATIONS ############
if config_variables['FLOW_SWITCH']['DATA_PREPROCESSING']:
    
    basket_data_prep = BasketDataProcessor(config_variables=config_variables)

    #Build Featureset
    if config_variables['DATA_PREPROCESS_PIPELINE']['CONTROL']['FEATURESET_BUILD']:
        
        basket_data_prep.build_featureset()

    #Get Plots to visualize correlation between variables
    if config_variables['DATA_PREPROCESS_PIPELINE']['CONTROL']['FEATURESET_CORRELATION']:

        basket_data_prep.get_correlation_plots()
    








