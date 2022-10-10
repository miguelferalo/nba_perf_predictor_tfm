from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl
import os
from datetime import date
import xlsxwriter
from time import sleep
import sys
from os.path import dirname, abspath


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
        

    def college_conferences_per_season(self): 

        'Save in a excel all the links for all coferences through selected seasons'

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

        'Save all the colleges urls for all the conferences in the input dictionary'

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

        'Fix problems with older collge data not having the same structure as the recent players'

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

        'Pull the data from the table'

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

        'Get all the possible data from college players'

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

        'Get all teh data from the players defined in the input list of URL'

        per_game_headers = ['id', 'name', 'position', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'school', 'conf', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '2p', '2pa', '2p%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 'orb', 'drb', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'unknown', 'strength_of_schedule']

        total_headers = ['id', 'name', 'position', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'school', 'conf', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '2p', '2pa', '2p%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 'orb', 'drb', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

        per_40_headers = ['id', 'name', 'position', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'school', 'conf', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '2p', '2pa', '2p%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

        per_100_headers = ['id', 'name', 'position', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'school', 'conf', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '2p', '2pa', '2p%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'unknown', 'offensive_rating', 'defensive_rating']

        advanced_headers = ['id', 'name', 'position', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year', 'season',
                            'school', 'conf', 'games', 'games_started', 'mp', 'per', 'ts%', 'efg%', '3par', 'ftr', 'pprod', 'orb%',
                            'drb%', 'trb%', 'ast%', 'stl%', 'blk%', 'tov%', 'usg%', 'unknown', 'ows', 'dws', 'ws', 'ws/40', 'unknown',
                            'obpm', 'dbpm', 'bpm']


        per_game_list = []
        total_list = []
        per_40_min_list = []
        per_100_poss_list = []
        advanced_list = []

        print('COLLEGE - Start scrapping data')

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

        'Load the file containing the links of the players we would like scrape'

        df = pd.read_excel(excel_path)

        links = df['player_url']

        return links

    def college_data_scraping(self):

        'Run a logic to scrape the data from university basketball players'

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

        return per_game_df, total_df, per_40_min_df, per_100_poss_df, advanced_df

    def get_nba_players_links(self):

        'Get a list with all the players in nba'

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

        return players_df['player_url']

    def nba_get_table_data(self, soup, table_id, general_data, output_list, pointer_3_year):

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

        'Get all the possible data from college players'

        global_url = self.config['SCRAPING_VARS']['NBA']['BASKERBALL_REFERENCE_BASE_URL']
        url = global_url + player_url

        html = urlopen(url)

        pointer_3_year = 1987
                            
        # create beautiful soup object from HTML
        soup = BeautifulSoup(html, features="html")

        #Get player info
        info = soup.find(id = 'info')
        player_name = info.find_all('span')[0].get_text()
        info_draft = [data.getText() for data in info.find_all('strong')]

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

        'Get all the data from the NBA players defined in the input list of URL'

        per_game_headers = ['id', 'name', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'age', 'team', 'league', 'position', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '3p', '3pa', '3p%', '2p', '2pa', '2p%', 'efg%', 'ft', 'fta', 'ft%', 'orb', 'drb', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

        total_headers = ['id', 'name', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'age', 'team', 'league', 'position', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '3p', '3pa', '3p%', '2p', '2pa', '2p%', 'efg%', 'ft', 'fta', 'ft%', 'orb', 'drb', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

        per_36_headers = ['id', 'name', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'age', 'team', 'league', 'position', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '3p', '3pa', '3p%', '2p', '2pa', '2p%', 'ft', 'fta', 'ft%', 'orb', 'drb', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts']

        per_100_headers = ['id', 'name', 'position', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year',
                            'season', 'age', 'team', 'league', 'position', 'games', 'games_started', 'mp', 'fg', 'fga', 'fg%',
                            '3p', '3pa', '3p%', '2p', '2pa', '2p%','ft', 'fta', 'ft%', 'tbr', 
                            'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'unknown', 'offensive_rating', 'defensive_rating']

        advanced_headers = ['id', 'name', 'heigth_cm', 'heigth_feet', 'draft', 'draft_team', 'draft_overall', 'draft_year', 'season',
                            'age', 'team', 'league', 'position', 'games', 'mp', 'per', 'ts%', '3par', 'ftr', 'orb%',
                            'drb%', 'trb%', 'ast%', 'stl%', 'blk%', 'tov%', 'usg%', 'unknown', 'ows', 'dws', 'ws', 'ws/48', 'unknown',
                            'obpm', 'dbpm', 'bpm', 'vorp']


        per_game_list = []
        total_list = []
        per_36_min_list = []
        per_100_poss_list = []
        advanced_list = []

        print('NBA - Start scrapping data')

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

        'scrape data from nba players'

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

        return per_game_df, total_df, per_36_min_df, per_100_poss_df, advanced_df
  