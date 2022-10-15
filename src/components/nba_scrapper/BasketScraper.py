from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import os
from os.path import dirname, abspath
import time


class BasketScrapper(object):

    def __init__(self, config_variables):
        self.config = config_variables
        self.college_year_start = config_variables['SCRAPING_VARS']['COLLEGE']['START_YEAR']
        self.college_year_end = config_variables['SCRAPING_VARS']['COLLEGE']['END_YEAR']
        self.nba_year_start = config_variables['SCRAPING_VARS']['NBA']['START_YEAR']
        self.nba_year_end = config_variables['SCRAPING_VARS']['NBA']['END_YEAR']
        self.data_folder = config_variables['FOLDERS']['DATA']
        self.url_folder = config_variables['FOLDERS']['URL']
        self.college_raw_folder = config_variables['FOLDERS']['COLLEGE_RAW']
        self.nba_raw_folder = config_variables['FOLDERS']['NBA_RAW']

        # Get Tables IDs from College players
        self.college_per_game_id = config_variables['SCRAPING_VARS']['COLLEGE']['TABLES_IDS']['PER_GAME']
        self.college_totals_id = config_variables['SCRAPING_VARS']['COLLEGE']['TABLES_IDS']['TOTALS']
        self.college_per_min_id = config_variables['SCRAPING_VARS']['COLLEGE']['TABLES_IDS']['PER_MINUTE']
        self.college_per_poss_id = config_variables['SCRAPING_VARS']['COLLEGE']['TABLES_IDS']['PER_POSS']
        self.college_advanced_id = config_variables['SCRAPING_VARS']['COLLEGE']['TABLES_IDS']['ADVANCED']

        # Get Tables IDs from NBA players
        self.nba_per_game_id = config_variables['SCRAPING_VARS']['NBA']['TABLES_IDS']['PER_GAME']
        self.nba_totals_id = config_variables['SCRAPING_VARS']['NBA']['TABLES_IDS']['TOTALS']
        self.nba_per_min_id = config_variables['SCRAPING_VARS']['NBA']['TABLES_IDS']['PER_MINUTE']
        self.nba_per_poss_id = config_variables['SCRAPING_VARS']['NBA']['TABLES_IDS']['PER_POSS']
        self.nba_advanced_id = config_variables['SCRAPING_VARS']['NBA']['TABLES_IDS']['ADVANCED']

        # Get Table Headers for College Players
        self.college_per_game_header = config_variables['SCRAPING_VARS']['COLLEGE']['HEADERS']['PER_GAME']
        self.college_totals_header = config_variables['SCRAPING_VARS']['COLLEGE']['HEADERS']['TOTALS']
        self.college_per_min_header = config_variables['SCRAPING_VARS']['COLLEGE']['HEADERS']['PER_40_MIN']
        self.college_per_poss_header = config_variables['SCRAPING_VARS']['COLLEGE']['HEADERS']['PER_100_POSS']
        self.college_advanced_header = config_variables['SCRAPING_VARS']['COLLEGE']['HEADERS']['ADVANCED']

        # Get Table Headers for NBA Players
        self.nba_per_game_header = config_variables['SCRAPING_VARS']['NBA']['HEADERS']['PER_GAME']
        self.nba_totals_header = config_variables['SCRAPING_VARS']['NBA']['HEADERS']['TOTALS']
        self.nba_per_min_header = config_variables['SCRAPING_VARS']['NBA']['HEADERS']['PER_36_MIN']
        self.nba_per_poss_header = config_variables['SCRAPING_VARS']['NBA']['HEADERS']['PER_100_POSS']
        self.nba_advanced_header = config_variables['SCRAPING_VARS']['NBA']['HEADERS']['ADVANCED']
                

    def college_conferences_per_season(self): 

        """
            Description:
                This function scrapes all the different conferences urls for each season specified at the input.

            Input Parameters:
                - config.yaml

            Return:
                - season_year : dictionary with all the conferences url per season.
                - Saves the dictionary in an excel file.
            
        """

        season_year = {}

        config_variables = self.config

        start_year = self.college_year_start
        end_year = self.college_year_end

        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        url_folder = self.url_folder
        excel_file = config_variables['SCRAPING_VARS']['COLLEGE']['CONFERENCES_PER_SEASON_EXCEL'].format(start_year = start_year,
                                                                                                            end_year = end_year)
        excel_path = os.path.join(top_dir, data_folder, url_folder, excel_file)
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 

        # Loop through each season to get all conferences
        for year in range(start_year,end_year):

            url = 'https://www.sports-reference.com/cbb/seasons/{year}.html'.format(year = str(year))

            # collect HTML data
            html = urlopen(url)
                    
            # create beautiful soup object from HTML
            soup = BeautifulSoup(html, features="html")

            # get table
            table = soup.find('table', id='conference-summary')

            # Collecting Data
            conference_data = []
            for row in table.tbody.find_all('tr'):    
                conference_data.append(row)

            rows_data = [[td.getText().replace('-', ' ') for td in conference_data[i].findAll('td')]
                                for i in range(len(conference_data))]

            conference_list =[]

            for conference in rows_data[0:]:
                conference_list.append(conference[0])

            hyperlinks_dict = {}
            for link in table.tbody.find_all('a'):
                if link.get_text() in conference_list:
                    hyperlinks_dict[link.get_text()] = link.get('href')

            # Add to dictionary
            season_year[str(year)] = hyperlinks_dict

            # Add to excel
            hyperlinks_df = pd.DataFrame(hyperlinks_dict.items(), columns=['Conference', 'URL'])
            hyperlinks_df.to_excel(writer, sheet_name=str(year), index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        return season_year

    def find_colleges_links(self, season_links):

        """
            Description:
                This function scrapes all the colleges url within the specified conferences and season
                at the input.

            Input Parameters:
                - config.yaml
                - season_links: dictionary with all the urls of conferences which need to be scraped

            Return:
                - season_links: dictionaty with all the links to colleges per season.
                - Saves the dictionary in an excel file.
            
        """

        for year in season_links:

            print('Scraping colleges for the {year} season'.format(year = year))

            colleges_dict = {}
            college_season_list = []
            college_url = []
            for conference in season_links[year]:

                url = season_links[year][conference]

                # collect HTML data
                root_link = self.config['SCRAPING_VARS']['COLLEGE']['COLLEGE_REFERENCE_BASE_URL']
                url = root_link + url
                html = urlopen(url)
                        
                # create beautiful soup object from HTML
                soup = BeautifulSoup(html, features="html")

                # get table
                table = soup.find('table', id='standings')

                # Collecting Data
                college_data = []
                for row in table.tbody.find_all('tr'):    
                    college_data.append(row)

                rows_data = [[td.getText().replace('-', ' ') for td in college_data[i].findAll('td')]
                                    for i in range(len(college_data))]

                college_list =[]

                for college in rows_data[0:]:
                    if(len(college) > 0):
                        college_list.append(college[0])

                hyperlinks_dict = {}
                for link in table.tbody.find_all('a'):
                    if link.get_text() in college_list:
                        hyperlinks_dict[link.get_text()] = link.get('href')

                        #Add to list
                        college_data = '{college}_{conference}_{year}'.format(college = link.get_text(),
                                                                                conference = conference, year = year) 

                        college_season_list.append(college_data)
                        college_url.append(link.get('href'))

                
                colleges_dict[conference] = hyperlinks_dict

            season_links[year]['college'] = colleges_dict

        #Save to excel
        end_year = self.college_year_end
        start_year = self.college_year_start
        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        url_folder = self.url_folder
        excel_file = self.config['SCRAPING_VARS']['COLLEGE']['COLLEGES_PER_SEASON_EXCEL'].format(start_year = start_year,
                                                                                                 end_year = end_year)
        excel_path = os.path.join(top_dir, data_folder, url_folder, excel_file)
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 

        colleges_df = pd.DataFrame()
        colleges_df['college_data'] = college_season_list
        colleges_df['college_url'] = college_url

        colleges_df.to_excel(writer)

        writer.save()

        return season_links

    def get_players_links(self, season_links):

        """
            Description:
                This function scrapes all the players ursl within the specified colleges and seasons
                at the input.

            Input Parameters:
                - config.yaml
                - season_links: dictionary with all the urls of colleges which need to be scraped.

            Return:
                - players_links: dictionaty with all the links to players per season.
                - Saves the dictionary in an excel file.
            
        """

        players_links = []
        player_names = []
        for year in season_links:

            print('Scraping Players links for {year} season'.format(year = year))

            for conferance in season_links[year]['college']:
                
                for college in season_links[year]['college'][conferance]:

                    url = season_links[year]['college'][conferance][college]

                    # collect HTML data
                    root_link = self.config['SCRAPING_VARS']['COLLEGE']['COLLEGE_REFERENCE_BASE_URL']
                    url = root_link + url
                    html = urlopen(url)    
                            
                    # create beautiful soup object from HTML
                    soup = BeautifulSoup(html, features="html")

                    # get table
                    table = soup.find('table', id='roster')

                    # Collecting Data
                    if table is not None:
                        for row in table.tbody.find_all('tr'):    
                            # Find all data for each column
                            link = row.find_all('a')[0]
                            
                            if link.get('href') not in players_links:
                                players_links.append(link.get('href'))
                                player_names.append(link.get_text())

                    else:
                        
                        print('The following link does not have the desirable table ' + url)

        #Save to excel
        end_year = self.college_year_end
        start_year = self.college_year_start
        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        url_folder = self.url_folder
        excel_file = self.config['SCRAPING_VARS']['COLLEGE']['PLAYERS_URLS_EXCEL'].format(start_year = start_year,
                                                                                                 end_year = end_year)
        excel_path = os.path.join(top_dir, data_folder, url_folder, excel_file)
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 

        players_df = pd.DataFrame()
        players_df['player_name'] = player_names
        players_df['player_url'] = players_links

        players_df.to_excel(writer)

        writer.save()

        return players_links

    def college_advanced_table_formatter(self, data_list, standard_length):

        """
            Description:
                The same table type for college data can have different columns depending on the season.
                Most recent seasons contain more data columns. This function tackles this problem and 
                make sure all the tables have the same structure.

            Input Parameters:
                - config.yaml
                - data_list: list containing the data of a season for a player
                - standard_length: int - the length that the row is supposed to have to standardize season data.

            Return:
                - data_formatted: list containing the player data for the season fixed to contain the correct number of values.
            
        """

        data_formatted = []
        for index, data in enumerate(data_list):

            data_formatted.append(data)

            # Get data to format
            if index == 13:
                data_formatted.append('-')
                
            if index == len(data_list) - 1:
                
                while(len(data_formatted)) < standard_length:
                    data_formatted.append('-')

        return data_formatted   

    def college_get_table_data(self, soup, table_id, general_data, output_list, pointer_3_year):

        """
            Description:
                Scrape all the college data required for a specific table.

            Input Parameters:
                - config.yaml
                - soup: beatifulsoup object containing all the data found at the given html.
                - table_id: the data type table to scrape the data (per_game, totals, per_36_minutes, per_100_possesions or advancded)
                - general_data: list containing data of the player shared across all tables such as Name or Height.
                - output_list: the specific data list where each row of information is stored (list of lists).
                - pointer_3_year: int - the first yeat where the 3 point line was implemented at college.

            Return:
                - output_list: the specific data list where each row of information is stored (list of lists).
            
        """

        # get table player_per game
        table = soup.find('table', id=table_id)

        # Collecting Data
        if table is not None:
            data = []
            for row in table.tbody.find_all('tr'):    
                data.append(row)
            
            for i in range(len(data)):
                get_season_year = int(data[i].find_all('a')[0].get_text().split('-')[0][0:2] + data[i].find_all('a')[0].get_text().split('-')[1])
                if get_season_year >= pointer_3_year:
                    general_data_season = general_data.copy()
                    general_data_season.append(get_season_year)
                    stats_data = [td.getText() for td in data[i].findAll('td')]
                    output_data = general_data_season + stats_data

                    if table_id == 'players_advanced':
                        standard_length = 38
                        if len(output_data) < standard_length:
                            output_data = self.college_advanced_table_formatter(output_data, standard_length)
                            
                    output_list.append(output_data)

        return output_list

    def player_data_college_reference(self, player_url, id, per_game_list, total_list, per_40_min_list, per_100_poss_list, advanced_list):

        """
            Description:
                Function to scrape all the data for the input player

            Input Parameters:
                - config.yaml
                - player_url: the url for the player whose data wants to be scraped.
                - id: the id given to the player. This id is unique for each player.
                - per_game_list: list where the per game data is stored.
                - total_list: list where the total data is stored.
                - per_40_min_list: list where the per 40 minutes data is stored.
                - per_100_poss_list: list where the per 100 possessions data is stored.
                - per_advanced_list40_min_list: list where the advanced data is stored.

            Return:
                - per_game_list: list where the per game data is stored.
                - total_list: list where the total data is stored.
                - per_40_min_list: list where the per 40 minutes data is stored.
                - per_100_poss_list: list where the per 100 possessions data is stored.
                - per_advanced_list40_min_list: list where the advanced data is stored.
            
        """

        global_url = self.config['SCRAPING_VARS']['COLLEGE']['COLLEGE_REFERENCE_BASE_URL']
        url = global_url + player_url

        html = urlopen(url)

        pointer_3_year = 1987
                            
        # create beautiful soup object from HTML
        soup = BeautifulSoup(html, features="html")

        #Get player info
        info = soup.find(id = 'info')
        player_name = info.find_all('span')[0].get_text()
        info_draft = [data.getText() for data in info.find_all('strong')]

        if len([data.get_text().replace('\n', '').replace(' ', '') for index, data in enumerate(info.find_all('p')[0])]) > 2:
            info_general = [data.get_text().replace('\n', '').replace(' ', '') for index, data in enumerate(info.find_all('p')[0])]
        else:
            info_general = [data.get_text().replace('\n', '').replace(' ', '') for index, data in enumerate(info.find_all('p')[1])]

        position = 'unknown'
        if 'Position:' in info_general:
            position_index = info_general.index('Position:') + 1
            position = info_general[position_index]
        elif len(info_general) < 2:
            print(id, url)
            

        height_cm = 'unknown'
        height_feet = 'unknown' 
        if 'cm' in ' '.join(info_general):
            for index in range(len(info_general)):
                if 'cm' in info_general[index]:
                    heigth_index =index
                    height_cm = info_general[heigth_index].split('(')[1].split(',')[0].replace('cm', '')[0:3]
                    height_feet = info_general[heigth_index].split()[0].replace(',', '')
                    break  
        elif len(info_general) < 2:
            print(id, url)
        
        if 'Draft:' in info_draft:
            draft = 1
            #Find which row contains draft
            for index, data in enumerate([data.get_text() for data in info.find_all('p')]):
                if 'Draft:' in data:
                    row = index
                    break
            team_draft = [data.get_text() for data in info.find_all('p')][index].split('Draft: ')[1].split(', ')[0]
            draft_overall = [data.get_text() for data in info.find_all('p')][index].split('Draft: ')[1].split(', ')[2].split(' ')[0][0:-2]
            draft_year = [data.get_text() for data in info.find_all('p')][index].split(', ')[-1].split(' ')[0]
        else:
            draft = 0
            team_draft = 'none'
            draft_overall = 0
            draft_year = 0

        general_data = [id, player_name, position, height_cm, height_feet, draft, team_draft, draft_overall, draft_year]


        # get table player_per game
        self.college_get_table_data(soup, self.college_per_game_id, general_data, per_game_list, pointer_3_year)

        # get table player totals
        self.college_get_table_data(soup, self.college_totals_id, general_data, total_list, pointer_3_year)

        # get table 40 mins
        self.college_get_table_data(soup, self.college_per_min_id, general_data, per_40_min_list, pointer_3_year)

        # get per 100 poss
        self.college_get_table_data(soup, self.college_per_poss_id, general_data, per_100_poss_list, pointer_3_year)

        # Advanced
        self.college_get_table_data(soup, self.college_advanced_id, general_data, advanced_list, pointer_3_year)

        return per_game_list, total_list, per_40_min_list, per_100_poss_list, advanced_list 

    def get_excel_raw_college(self, players_links):

        """
            Description:
                Function to get all the data for all the players specified at the input

            Input Parameters:
                - config.yaml
                - player_links: list of urls with all the players that are required to scrape.


            Return:
                - per_game_df: pandas dataframe containing all the players and seasons per game data.
                - total_df: pandas dataframe containing all the players and seasons total data.
                - per_40_min_df: pandas dataframe containing all the players and seasons per 40 minutes data.
                - per_100_poss_df: pandas dataframe containing all the players and seasons per 100 possessions data.
                - advanced_df: pandas dataframe containing all the players and seasons advanced data.  
                - Save all dataframes in one excel file (multiple sheets).       
        """

        # Get dataframes headers
        per_game_headers = self.college_per_game_header
        total_headers = self.college_totals_header
        per_40_headers = self.college_per_min_header
        per_100_headers = self.college_per_poss_header
        advanced_headers = self.college_advanced_header


        per_game_list = []
        total_list = []
        per_40_min_list = []
        per_100_poss_list = []
        advanced_list = []

        print('COLLEGE - Start scraping data')

        total_players = len(players_links)
        percentage_status = 0.10
        every_players_status = round(total_players * percentage_status)

        for id, link in enumerate(players_links):
            try:

                if id % every_players_status == 0:
                    percentage = round(id / every_players_status * percentage_status * 100)
                    print('{number}% of the total players have been scrapped'.format(number = str(percentage)))

                formatted_id = f'{id:05d}'
                self.player_data_college_reference(link, formatted_id, per_game_list, total_list, per_40_min_list, per_100_poss_list, advanced_list)
            except:
                print(id, link)

        print('COLLEGE - SCRAPING DATA DONE')

        print('COLLEGE - PER GAME TO DF START')
        per_game_df =pd.DataFrame(per_game_list, columns = per_game_headers)
        print('COLLEGE - PER GAME TO DF DONE')

        print('COLLEGE - TOTAL TO DF START')
        total_df =pd.DataFrame(total_list, columns = total_headers)
        print('COLLEGE - TOTAL TO DF DONE')

        print('COLLEGE - 40 MIN TO DF START')
        per_40_min_df =pd.DataFrame(per_40_min_list, columns = per_40_headers)
        print('COLLEGE - 40 MIN TO DF DONE')

        print('COLLEGE - PER 100 POSS TO DF START')
        per_100_poss_df =pd.DataFrame(per_100_poss_list, columns = per_100_headers)
        print('COLLEGE - PER 100 POSS TO DF DONE')

        print('COLLEGE - ADVANCED TO DF START')
        advanced_df =pd.DataFrame(advanced_list, columns = advanced_headers)
        print('COLLEGE - ADVANCED TO DF DONE')

        # Convert to excel
        print('COLLEGE - START TO EXCEL')
        # Create a Pandas Excel writer using XlsxWriter as the engine.

        end_year = self.college_year_end
        start_year = self.college_year_start
        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        college_raw_data_folder = self.college_raw_folder
        excel_file = self.config['SCRAPING_VARS']['COLLEGE']['PLAYERS_RAW_DATA_EXCEL'].format(start_year = start_year,
                                                                                                 end_year = end_year)

        excel_path = os.path.join(top_dir, data_folder, college_raw_data_folder, excel_file)
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 

        # Write each dataframe to a different worksheet.
        per_game_df.to_excel(writer, sheet_name='PER_GAME_DATA', index=False)
        total_df.to_excel(writer, sheet_name='SEASON_TOTAL_DATA', index=False)
        per_40_min_df.to_excel(writer, sheet_name='PER_40_MIN_DATA', index=False)
        per_100_poss_df.to_excel(writer, sheet_name='PER_100_POSS_DATA', index=False)
        advanced_df.to_excel(writer, sheet_name='SEASON_ADVANCED_DATA', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        print('COLLEGE - TO EXCEL DONE')

        return per_game_df, total_df, per_40_min_df, per_100_poss_df, advanced_df

    def load_player_links(self, excel_path):

        """
            Description:
                Function to load the players links from an excel file

            Input Parameters:
                - config.yaml
                - excel_path: path to the excel which contains the players urls.


            Return:
                - links: list with all the urls from the players      
        """

        df = pd.read_excel(excel_path)

        links = df['player_url']

        return links

    def college_data_scraping(self):

        """
            Description:
                Function to scrape college players data from College Reference.

            Input Parameters:
                - config.yaml

            Return:
                - per_game_df: pandas dataframe containing all the players and seasons per game data.
                - total_df: pandas dataframe containing all the players and seasons total data.
                - per_40_min_df: pandas dataframe containing all the players and seasons per 40 minutes data.
                - per_100_poss_df: pandas dataframe containing all the players and seasons per 100 possessions data.
                - advanced_df: pandas dataframe containing all the players and seasons advanced data.        
        """

        start_time = time.time()

        #Check if there is already a file with the players links
        end_year = self.college_year_end
        start_year = self.college_year_start
        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        url_folder = self.url_folder
        excel_file = self.config['SCRAPING_VARS']['COLLEGE']['PLAYERS_URLS_EXCEL'].format(start_year = start_year,
                                                                                                 end_year = end_year)
        excel_path = os.path.join(top_dir, data_folder, url_folder, excel_file)

        if os.path.exists(excel_path):
            
            #Load excel file
            player_links = self.load_player_links(excel_path)

        else:
        
            #Get URLs for each conference each year
            season_links = self.college_conferences_per_season()

            #Get URLs for each conference each year
            season_links = self.find_colleges_links(season_links)

            #Get URLs for each player each year
            player_links = self.get_players_links(season_links)

        #Scrape the players data
        per_game_df, total_df, per_40_min_df, per_100_poss_df, advanced_df = self.get_excel_raw_college(player_links)

        #Print Run time
        print("{step} took {time} minutes".format(step = 'The College Data web scraping step', time = round((time.time() - start_time) / 60, 2)))

        return per_game_df, total_df, per_40_min_df, per_100_poss_df, advanced_df

    def get_nba_players_links(self):

        """
            Description:
                This function scrapes all the NBA players urls specified in the input parameters.

            Input Parameters:
                - config.yaml.

            Return:
                - players_links: dictionaty with all the links to players per season.
                - Saves the dictionary in an excel file.
            
        """

        root_url = self.config['SCRAPING_VARS']['NBA']['BASKERBALL_REFERENCE_BASE_URL']

        start_year = self.nba_year_start
        end_year = self.nba_year_end

        player_dict = {}

        for year in range(start_year, end_year):

            print('Scraping Players links for {year} season'.format(year = year))

            season_url = self.config['SCRAPING_VARS']['NBA']['NBA_LEAGUE_PER_GAME_STATS'] 
            url = root_url + season_url.format(year = year)

            # collect HTML data
            html = urlopen(url)    
                    
            # create beautiful soup object from HTML
            soup = BeautifulSoup(html, features="html")

            # get table
            table = soup.find('table', id=self.config['SCRAPING_VARS']['NBA']['PER_GAME_TABLE_ID'])

            # Collecting Data
            if table is not None:
                for row in table.tbody.find_all('tr'):    
                    # Find if the row has a link - player
                    if len(row.find_all('a')) > 0:
                        link = row.find_all('a')[0]

                        if link.get('href') not in player_dict.keys():
                            player_dict[link.get('href')] = link.get_text()
            else:
                
                print('The following link does not have the desirable table ' + url)

        #Save to excel
        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        url_folder = self.url_folder
        excel_file = self.config['SCRAPING_VARS']['NBA']['PLAYERS_URLS_EXCEL'].format(start_year = start_year,
                                                                                                 end_year = end_year)
        excel_path = os.path.join(top_dir, data_folder, url_folder, excel_file)
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 

        players_df = pd.DataFrame(player_dict.items(), columns=['player_url', 'player_name'])

        players_df.to_excel(writer, index = False)

        writer.save()

        players_links = players_df['player_url']

        return players_links

    def nba_get_table_data(self, soup, table_id, general_data, output_list, pointer_3_year):

        """
            Description:
                Get all the data from a specific data type table.

            Input Parameters:
                - config.yaml
                - soup: beatifulsoup object containing all the data found at the given html.
                - table_id: the data type table to scrape the data (per_game, totals, per_36_minutes, per_100_possesions or advancded)
                - general_data: list containing data of the player shared across all tables such as Name or Height.
                - output_list: the specific data list where each row of information is stored (list of lists).
                - pointer_3_year: int - the first yeat where the 3 point line was implemented at college.

            Return:
                - output_list: the specific data list where each row of information is stored (list of lists).       
        """

        'Pull the data from the table'

        # get table player_per game
        table = soup.find('table', id=table_id)

        # Collecting Data
        if table is not None:
            data = []
            for row in table.tbody.find_all('tr'):    
                data.append(row)
            
            for i in range(len(data)):
                get_season_year = int(data[i].get_text().replace(' ', '')[0:2] + data[i].get_text().replace(' ', '')[5:7])
                if get_season_year >= pointer_3_year:
                    general_data_season = general_data.copy()
                    general_data_season.append(get_season_year)
                    stats_data = [td.getText() for td in data[i].findAll('td')]
                    output_data = general_data_season + stats_data

                    if (table_id == self.nba_totals_id) & (len(output_data) > 38):
                        output_data = output_data[0:-2]

                    output_list.append(output_data)

        return output_list

    def player_data_nba_reference(self, player_url, id, per_game_list, total_list, per_40_min_list, per_100_poss_list, advanced_list):

        """
            Description:
                Function to get scrape all the data form the input NBA player

            Input Parameters:
                - config.yaml
                - player_url: url of the NBA player
                - id: int - unique id for the NBA player
                - per_game_list: list containing all the per game data for the player.
                - total_list: list containing all the total data for the player.
                - per_40_min_list: list containing all the per 40 minutes data for the player.
                - per_100_poss_list: list containing all the per 100 possessions data for the player.
                - advanced_list: list containing all the advanced data for the player.

            Return:
                - per_game_list: list containing all the per game data for the player.
                - total_list: list containing all the total data for the player.
                - per_40_min_list: list containing all the per 40 minutes data for the player.
                - per_100_poss_list: list containing all the per 100 possessions data for the player.
                - advanced_list: list containing all the advanced data for the player.       
        """

        global_url = self.config['SCRAPING_VARS']['NBA']['BASKERBALL_REFERENCE_BASE_URL']
        url = global_url + player_url

        html = urlopen(url)

        pointer_3_year = 1987
                            
        # create beautiful soup object from HTML
        soup = BeautifulSoup(html, features="html")

        #Get player info
        info = soup.find(id = 'info')
        player_name = info.find_all('span')[0].get_text()

        info_general = [data.get_text().replace('\n', '').replace(' ', '') for index, data in enumerate(info.find_all('p'))]

        height_cm = 'unknown'
        height_feet = 'unknown'

        draft = 0
        team_draft = 'none'
        draft_overall = 0
        draft_year = 0

        #Find which row contains cm and kg
        for index, data in enumerate(info_general):
            if 'cm' in data:
                height_cm = data.split('(')[1].split('cm')[0]
                height_feet = data.split(',')[0]
        
            if 'Draft:' in data:
                draft = 1
                team_draft = data.split('Draft:')[1].split(',')[0]
                draft_overall = data.split('(')[1].split(',')[1].split('overall')[0][0:-2]
                draft_year = data.split(',')[-1].split('NBA')[0]

        general_data = [id, player_name, height_cm, height_feet, draft, team_draft, draft_overall, draft_year]


        # get table player_per game
        self.nba_get_table_data(soup, self.nba_per_game_id, general_data, per_game_list, pointer_3_year)

        # get table player totals
        self.nba_get_table_data(soup, self.nba_totals_id, general_data, total_list, pointer_3_year)

        # get table 36 mins
        self.nba_get_table_data(soup, self.nba_per_min_id, general_data, per_40_min_list, pointer_3_year)

        # get per 100 poss
        self.nba_get_table_data(soup, self.nba_per_poss_id, general_data, per_100_poss_list, pointer_3_year)

        # Advanced
        self.nba_get_table_data(soup, self.nba_advanced_id, general_data, advanced_list, pointer_3_year)

        return per_game_list, total_list, per_40_min_list, per_100_poss_list, advanced_list
    
    def get_excel_raw_nba(self, players_links):

        """
            Description:
                Function to scrape NBA players data from Basketball Reference.

            Input Parameters:
                - config.yaml
                - players_links: list containing all the NBA players urls to be scrapped.

            Return:
                - per_game_df: pandas dataframe containing all the players and seasons per game data.
                - total_df: pandas dataframe containing all the players and seasons total data.
                - per_36_min_df: pandas dataframe containing all the players and seasons per 40 minutes data.
                - per_100_poss_df: pandas dataframe containing all the players and seasons per 100 possessions data.
                - advanced_df: pandas dataframe containing all the players and seasons advanced data.  
                - Save all the dataframes in one excel (multiple sheets).      
        """

        #Get NBA dataframes headers
        per_game_headers = self.nba_per_game_header
        total_headers = self.nba_totals_header
        per_36_headers = self.nba_per_min_header
        per_100_headers = self.nba_per_poss_header
        advanced_headers = self.nba_advanced_header


        per_game_list = []
        total_list = []
        per_36_min_list = []
        per_100_poss_list = []
        advanced_list = []

        print('NBA - Start scraping data')

        total_players = len(players_links)
        percentage_status = 0.10
        every_players_status = round(total_players * percentage_status)

        for id, link in enumerate(players_links):
            try:

                if id % every_players_status == 0:
                    percentage = round(id / every_players_status * percentage_status * 100)
                    print('{number}% of the total players have been scrapped'.format(number = str(percentage)))

                formatted_id = f'{id:05d}'
                self.player_data_nba_reference(link, formatted_id, per_game_list, total_list, per_36_min_list, per_100_poss_list, advanced_list)
            except:
                print(id, link)

        print('NBA - SCRAPING DATA DONE')

        print('NBA - PER GAME TO DF START')
        per_game_df =pd.DataFrame(per_game_list, columns = per_game_headers)
        print('NBA - PER GAME TO DF DONE')

        print('NBA - TOTAL TO DF START')
        total_df =pd.DataFrame(total_list, columns = total_headers)
        print('NBA - TOTAL TO DF DONE')

        print('NBA - 36 MIN TO DF START')
        per_36_min_df =pd.DataFrame(per_36_min_list, columns = per_36_headers)
        print('NBA - 36 MIN TO DF DONE')

        print('NBA - PER 100 POSS TO DF START')
        per_100_poss_df =pd.DataFrame(per_100_poss_list, columns = per_100_headers)
        print('NBA - PER 100 POSS TO DF DONE')

        print('NBA - ADVANCED TO DF START')
        advanced_df =pd.DataFrame(advanced_list, columns = advanced_headers)
        print('NBA - ADVANCED TO DF DONE')

        # Convert to excel
        print('NBA - START TO EXCEL')
        # Create a Pandas Excel writer using XlsxWriter as the engine.

        end_year = self.nba_year_end
        start_year = self.nba_year_start
        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        nba_raw_data_folder = self.nba_raw_folder
        excel_file = self.config['SCRAPING_VARS']['NBA']['PLAYERS_RAW_DATA_EXCEL'].format(start_year = start_year,
                                                                                                 end_year = end_year)

        excel_path = os.path.join(top_dir, data_folder, nba_raw_data_folder, excel_file)
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 

        # Write each dataframe to a different worksheet.
        per_game_df.to_excel(writer, sheet_name='PER_GAME_DATA', index=False)
        total_df.to_excel(writer, sheet_name='SEASON_TOTAL_DATA', index=False)
        per_36_min_df.to_excel(writer, sheet_name='PER_36_MIN_DATA', index=False)
        per_100_poss_df.to_excel(writer, sheet_name='PER_100_POSS_DATA', index=False)
        advanced_df.to_excel(writer, sheet_name='SEASON_ADVANCED_DATA', index=False)

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        print('NBA - TO EXCEL DONE')

        return per_game_df, total_df, per_36_min_df, per_100_poss_df, advanced_df

    def nba_data_scraping(self):

        """
            Description:
                Function to scrape NBA players data from Basketball Reference.

            Input Parameters:
                - config.yaml

            Return:
                - per_game_df: pandas dataframe containing all the players and seasons per game data.
                - total_df: pandas dataframe containing all the players and seasons total data.
                - per_36_min_df: pandas dataframe containing all the players and seasons per 40 minutes data.
                - per_100_poss_df: pandas dataframe containing all the players and seasons per 100 possessions data.
                - advanced_df: pandas dataframe containing all the players and seasons advanced data.        
        """

        start_time = time.time()

        #Check if there is already a file with the players links
        end_year = self.nba_year_end
        start_year = self.nba_year_start
        top_dir = dirname(dirname(dirname(dirname(abspath(__file__))))) 
        data_folder = self.data_folder
        url_folder = self.url_folder
        excel_file = self.config['SCRAPING_VARS']['NBA']['PLAYERS_URLS_EXCEL'].format(start_year = start_year,
                                                                                                 end_year = end_year)
        excel_path = os.path.join(top_dir, data_folder, url_folder, excel_file)

        if os.path.exists(excel_path):
            
            #Load excel file
            player_links = self.load_player_links(excel_path)

        else:
        
            #Scrape the players data
            player_links = self.get_nba_players_links()

        #Scrape the players data
        per_game_df, total_df, per_36_min_df, per_100_poss_df, advanced_df = self.get_excel_raw_nba(player_links)

        #Print Run time
        print("{step} took {time} minutes".format(step = 'The NBA Data web scraping step', time = round((time.time() - start_time) / 60, 2)))


        return per_game_df, total_df, per_36_min_df, per_100_poss_df, advanced_df
  