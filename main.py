from config import config_variables
from src.components.nba_scrapper.BasketScraper import BasketScrapper

######################## SET A WORKFLOW IN THE PROJECT #################################


if config_variables['FLOW_SWITCH']['WEB_SCRAPPING']:

    basketball_scrapper = BasketScrapper(config_variables=config_variables)

    if config_variables['FLOW_SWITCH']['COLLEGE_SCRAPPING']:

        #Get Data of college players
        basketball_scrapper.college_data_scraping()

    if config_variables['FLOW_SWITCH']['NBA_SCRAPING']:
        #Get Data of NBA players
        basketball_scrapper.nba_data_scraping()






