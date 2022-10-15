from config import config_variables
from src.components.nba_scrapper.BasketScraper import BasketScrapper
from src.components.data_preprocessing.BasketballDataProcessor import BasketDataProcessor
from src.components.modeling.DraftForecast import DraftForecast

######################## SET A WORKFLOW IN THE PROJECT #################################

############ PIPELINE TO SCRAPE DATA ############
if config_variables['FLOW_SWITCH']['WEB_SCRAPING']:

    print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'WEB SCRAPING PIPELINE'))
    basketball_scrapper = BasketScrapper(config_variables=config_variables)

    if config_variables['SCRAPING_VARS']['CONTROL']['COLLEGE_SCRAPING']:

        #Get Data of college players
        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'COLLEGE DATA WEB SCRAPING'))
        basketball_scrapper.college_data_scraping()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'COLLEGE DATA WEB SCRAPING'))

    if config_variables['SCRAPING_VARS']['CONTROL']['NBA_SCRAPING']:
        #Get Data of NBA players
        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'NBA DATA WEB SCRAPING'))
        basketball_scrapper.nba_data_scraping()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'NBA DATA WEB SCRAPING'))

    print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'WEB SCRAPING PIPELINE'))

############ PIPELINE TO CREATE FEATURESET, IMPUTE, VISUALIZE CORRELATIONS ############
if config_variables['FLOW_SWITCH']['DATA_PREPROCESSING']:

    print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'DATA PROCESSING PIPELINE'))
    basket_data_prep = BasketDataProcessor(config_variables=config_variables)

    #Build Featureset
    if config_variables['DATA_PREPROCESS_PIPELINE']['CONTROL']['FEATURESET_BUILD']:
        
        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'FEATURE SET BUILD'))
        basket_data_prep.build_featureset()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'FEATURE SET BUILD'))

    #Get Plots to visualize correlation between variables
    if config_variables['DATA_PREPROCESS_PIPELINE']['CONTROL']['FEATURESET_CORRELATION']:

        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'PEARSON CORRELATION FEATURE SET'))
        basket_data_prep.get_correlation_plots()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'PEARSON CORRELATION FEATURE SET'))

    #Visualize Missing Data columns for imputation purposes
    if config_variables['DATA_PREPROCESS_PIPELINE']['CONTROL']['FEATURESET_IMPUTATION_VISUALIZER']:

        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'DATA VISUALIZER IMPUTATION'))
        basket_data_prep.data_visualizer_imputation()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'DATA VISUALIZER IMPUTATION'))

    #Get a summary of the definitive dataframe
    if config_variables['DATA_PREPROCESS_PIPELINE']['CONTROL']['FEATURESET_VAR_ANALYSIS']:

        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'FEATURE SET SUMMARY'))
        basket_data_prep.definitive_featureset_summary()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'FEATURE SET SUMMARY'))

    print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'DATA PROCESSING PIPELINE'))
    
############ PIPELINE TO HYPERPARAMETER TUNE, TRAIN AND PREDICT ############
if config_variables['FLOW_SWITCH']['MODELINING']:
    
    print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'MODELING PIPELINE'))
    draft_forecast_wizard = DraftForecast(config_variables=config_variables)

    #Perform Hyperparameter tuning process
    if config_variables['MODELING_PIPELINE']['CONTROL']['HYPER_PARAMETER_TUNING']:
        
        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'HYPERPARAMETER TUNING'))
        draft_forecast_wizard.hyper_parameter_tuning()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'HYPERPARAMETER TUNING'))

    #Run hyperparameter diagnosis
    if config_variables['MODELING_PIPELINE']['CONTROL']['HYPER_PARAMETER_SUMMARY']:

        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'HYPERPARAMETER TUNING SUMMARY'))
        draft_forecast_wizard.hyperparameter_tuning_diagnosis()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'HYPERPARAMETER TUNING SUMMARY'))        

    #Train and Make Predictions
    if config_variables['MODELING_PIPELINE']['CONTROL']['TRAIN_AND_PREDICT']:

        print(config_variables['LOGGING_MESSAGE']['START'].format(step = 'TRAINING, PREDICTIONS AND EVALUATION'))
        draft_forecast_wizard.train_and_predict()
        print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'TRAINING, PREDICTIONS AND EVALUATIONR'))

    print(config_variables['LOGGING_MESSAGE']['END'].format(step = 'MODELING PIPELINE'))
