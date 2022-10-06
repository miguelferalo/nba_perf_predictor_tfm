import pandas as pd
import openpyxl
import os
from datetime import date
import sys
from os.path import dirname, abspath
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BasketDataProcessor(object):

    def __init__(self, config_variables):
        self.config = config_variables
        self.college_year_start = config_variables['SCRAPING_VARS']['COLLEGE']['START_YEAR']
        self.college_year_end = config_variables['SCRAPING_VARS']['COLLEGE']['END_YEAR']
        self.nba_year_start = config_variables['SCRAPING_VARS']['NBA']['START_YEAR']
        self.nba_year_end = config_variables['SCRAPING_VARS']['NBA']['END_YEAR']
        
        self.top_path = dirname(dirname(dirname(dirname(abspath(__file__)))))

        self.data_folder = config_variables['FOLDERS']['DATA']
        self.url_folder = config_variables['FOLDERS']['URL']
        self.college_raw_folder = config_variables['FOLDERS']['COLLEGE_RAW']
        self.nba_raw_folder = config_variables['FOLDERS']['NBA_RAW']
        self.featureset_folder = config_variables['FOLDERS']['FEATURESET']
        self.correlation_folder = config_variables['FOLDERS']['CORRELATION']

        self.college_raw_excel = config_variables['SCRAPING_VARS']['COLLEGE']['PLAYERS_RAW_DATA_EXCEL'].format(start_year = self.college_year_start, end_year = self.college_year_end)
        self.nba_raw_excel = config_variables['SCRAPING_VARS']['NBA']['PLAYERS_RAW_DATA_EXCEL'].format(start_year = self.nba_year_start, end_year = self.nba_year_end)

        self.nba_success_var = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['NBA']['NBA_SUCCESS_VAR']
        self.success_var_agg = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['NBA']['SUCCESS_AGG']
        self.year_build_start = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['YEAR_BUILD']
        self.rookie_contract_length = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['NBA']['ROOKIE_CONTRACT']

        self.college_per_game_table = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['TABLE_TYPES']['PER_GAME']
        self.college_totals_table = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['TABLE_TYPES']['TOTALS']
        self.college_per_40m_table = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['TABLE_TYPES']['PER_MINUTE']
        self.college_per_100p_table = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['TABLE_TYPES']['PER_POSS']
        self.college_advanced_table = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['TABLE_TYPES']['ADVANCED']

        self.college_tables_load = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['TABLES_LOAD']

        self.nba_advanced_table = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['NBA']['TABLE_TYPES']['ADVANCED']

        self.college_merging_columns = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['MERGING_COLUMNS']

        self.college_spec_data = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['SPEC_DATA']

        self.output_featureset_excel = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['FEATURESET_EXCEL'].format(output = self.nba_success_var, build_year = self.year_build_start)
        self.output_featureset_excel_preprocessed = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['FEATURESET_EXCEL_PREPROCESSED'].format(output = self.nba_success_var, build_year = self.year_build_start)    

        self.max_vars_correlation_plot = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['MAX_VARS_PLOT']
        self.success_var_folder = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['FOLDER']['SUCCESS_VAR'].format(var = self.nba_success_var.upper())
        self.correlation_drop_vars = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['NOT_RELEVANT_COLUMNS']
        self.high_correlation_threshold = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['HIGH_CORR_THRESHOLD']

        self.plot_1_title = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_1_TITLE'].format(var = self.nba_success_var.upper())
        self.plot_3_title = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_3_TITLE']
        self.plot_1_file = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_1_FILE']
        self.plot_2_file = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_2_FILE']  
        self.plot_3_file = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_3_FILE']         

        self.norm_var = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['NORM_VAR']
        self.vars_to_norm = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['NORMALIZED_FEATURES']
        self.outlier_vars = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['OUTLIER_DETECTOR_VARS']

        self.null_column__drop = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['NULL_COLUMNS_DROP']
        

    def nba_df_filter(self):

        path_to_nba_excel = os.path.join(self.top_path, self.data_folder, self.nba_raw_folder, self.nba_raw_excel)

        # Load df
        nba_df = pd.read_excel(path_to_nba_excel, sheet_name = self.nba_advanced_table)

        nba_df_rookie = nba_df[((nba_df['season'] - nba_df['draft_year']) <= self.rookie_contract_length + 1) &
                                (nba_df['draft_year'] > self.year_build_start) & 
                                (nba_df['draft'] == 1)].reset_index(drop = True).copy()

        if self.success_var_agg == 'mean':
            
            nba_df_rookie = nba_df_rookie.groupby(['id', 'draft_overall', 'draft_year', 'name', 'position', 'heigth_cm'], as_index=False)[self.nba_success_var].mean()
            nba_df_rookie.columns = ['id', 'draft_overall', 'draft_year', 'name', 'position', 'heigth_cm', '{success_var}_{agg}'.format(success_var = self.nba_success_var, agg = self.success_var_agg)]
        else:
            nba_df_rookie = nba_df_rookie.groupby(['id', 'draft_overall', 'draft_year', 'name', 'position', 'heigth_cm'], as_index=False)[self.nba_success_var].sum()
            nba_df_rookie.columns = ['id', 'draft_overall', 'draft_year', 'name', 'position', 'heigth_cm', '{success_var}_{agg}'.format(success_var = self.nba_success_var, agg = self.success_var_agg)]


        return nba_df_rookie

    def college_specific_data_columns(self, college_last_df, college_df, columns_spec):

        # Load df
        college_last_df = college_last_df.drop_duplicates(subset=['id'], keep='last')

        for column in columns_spec:

            if 'n_seasons' in column:

                n_season = college_df.groupby(['id'], as_index=False)['season'].count()
                n_season.columns = ['id', 'n_seasons']

                college_last_df = college_last_df.merge(n_season, on=['id'],
                                                                how='left')  

            if '(dev)' in column:

                spec_column = column.split(' (dev)')[0]

                spec_last_season = college_df.groupby(['id'], as_index=False).last('season')[['id', spec_column]]
                spec_first_season = college_df.groupby(['id'], as_index=False).first('season')[['id', spec_column]]


                spec_improvement_df  = spec_last_season.merge(spec_first_season, on=['id'], how='left')

                spec_improvement_df['{spec} (dev)'.format(spec = spec_column)] = spec_improvement_df['{spec}_x'.format(spec = spec_column)] - spec_improvement_df['{spec}_y'.format(spec = spec_column)]

                college_last_df = college_last_df.merge(spec_improvement_df[['id', '{spec} (dev)'.format(spec = spec_column)]], on=['id'],
                                                                how='left')

            if '(prop)' in column:

                spec_column = column.split(' (prop)')[0]

                school_df = college_df.groupby(['school', 'season'], as_index=False)[spec_column].sum()

                left_df_college = college_last_df.merge(school_df, on=['school', 'season'],
                                                                how='left', suffixes=('_left', '_right'))

                college_last_df['{spec} (prop)'.format(spec = spec_column)] = left_df_college['{spec}_left'.format(spec = spec_column)] / left_df_college['{spec}_right'.format(spec = spec_column)]

        return college_last_df

    def college_df_loader(self, data_type):

        specific_vars = self.college_spec_data[data_type]

        # Load df
        path_to_college_excel = os.path.join(self.top_path, self.data_folder, self.college_raw_folder, self.college_raw_excel)
        college_df = pd.read_excel(path_to_college_excel, sheet_name= data_type)

        # Get only data greater than year of study
        college_df = college_df[college_df['season'] >= self.year_build_start]

        # Get only the year previous to nba
        if 'ws' in college_df.columns:
            college_df['ws'] = college_df['ws'].replace('-', '0')
            college_df['ws'] = college_df['ws'].astype(float)
        college_df_last = college_df.groupby(['id','name', 'school'], as_index=False).last('season')

        # Get specific column types
        college_df_last = self.college_specific_data_columns(college_df_last, college_df, specific_vars)

        return college_df_last



    def college_filter_dataframe(self):

        college_dataframe_list = []

        df_to_load_list = self.college_tables_load 

        unchanged_column_names = self.college_merging_columns

        if self.college_per_game_table in df_to_load_list:

            college_per_game_df = self.college_df_loader(self.college_per_game_table)

            college_per_game_df.columns = ['{column} ({type})'.format(column = column, type = self.college_per_game_table.lower().split('_')[1]) if column not in unchanged_column_names else column for column in college_per_game_df]

            college_dataframe_list.append(college_per_game_df)

        if self.college_totals_table in df_to_load_list:

            college_totals_df = self.college_df_loader(self.college_totals_table)

            college_totals_df.columns = ['{column} ({type})'.format(column = column, type = self.college_totals_table.lower().split('_')[1]) if column not in unchanged_column_names else column for column in college_totals_df]

            college_dataframe_list.append(college_totals_df)

        if self.college_per_40m_table in df_to_load_list:

            college_per_40_m_df = college_df_loader(self.college_per_40_m_table)

            college_per_40_m_df.columns = ['{column} ({type})'.format(column = column, type = self.college_per_40m_table.lower().split('_')[2]) if column not in unchanged_column_names else column for column in college_per_40_m_df]

            college_dataframe_list.append(college_per_40_m_df)

        if self.college_per_100p_table in df_to_load_list:

            college_per_100_p_df = self.college_df_loader(self.college_per_100p_table)

            college_per_100_p_df.columns = ['{column} ({type})'.format(column = column, type = self.college_per_100p_table.lower().split('_')[2]) if column not in unchanged_column_names else column for column in college_per_100_p_df]

            college_dataframe_list.append(college_per_100_p_df)


        if self.college_advanced_table in df_to_load_list:

            college_advanced_df = self.college_df_loader(self.college_advanced_table)

            college_advanced_df.columns = ['{column} ({type})'.format(column = column, type = self.college_advanced_table.lower().split('_')[1]) if column not in unchanged_column_names else column for column in college_advanced_df]

            college_dataframe_list.append(college_advanced_df)

        # Merge all colleges dataframes
        college_global_df = pd.DataFrame()
        for index, dataframe in enumerate(college_dataframe_list):

            if index == 0:
                college_global_df = dataframe
            else:
                college_global_df = college_global_df.merge(dataframe, on = unchanged_column_names, how = 'left')

        return college_global_df
        

    def merge_predictor_college(self):

        # Get nba dataframe filtered
        nba_drafted_df = self.nba_df_filter()

        # Get college dataframe filtered
        college_filtered_df = self.college_filter_dataframe()

        # Merge to college
        nba_college_merge_df = nba_drafted_df.merge(college_filtered_df.drop(columns=['name', 'id']), on = ['draft_overall', 'draft_year'], how = 'left')

        # Remove players who did not go to college such as Lebron or Tracy McGrady
        nba_college_merge_df = nba_college_merge_df[nba_college_merge_df['season'].notnull()].reset_index(drop = True).copy()

        # Save to excel
        self.save_feature_set(nba_college_merge_df, self.output_featureset_excel)

        return nba_college_merge_df

    def norm_features(self, feature_set):

        # Get normalizing var and normalize it from 0 to 1
        normalizer =(feature_set[self.norm_var] - feature_set[self.norm_var].min())/(feature_set[self.norm_var].max() - feature_set[self.norm_var].min())

        for column in self.vars_to_norm:

            feature_set['{column} (normalized)'.format(column = column)] = feature_set[column] * normalizer

        return feature_set

    def outlier_corrector(self, featureset):

        # Get variables to correct outliers
        vars_to_correct = []
        for var_outlier, var_treshold, threshold in self.outlier_vars:

            # Find people below the theshold
            mean_var = featureset[featureset[var_treshold] < threshold][var_outlier].mean()
            std_var = featureset[featureset[var_treshold] < threshold][var_outlier].std()

            std_limit = mean_var + 2 * std_var

            # Find people below the theshold
            var_corrected = []
            for index, row in featureset.iterrows():

                if (row[var_treshold] < threshold) & (row[var_outlier] > std_limit):

                    var_corrected.append(mean_var)

                else:
                    var_corrected.append(row[var_outlier])

            
            featureset[var_outlier] = var_corrected

        return featureset

    def build_featureset(self):

        excel_path = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_excel)

        if os.path.exists(excel_path):

            nba_college_merge_df = pd.read_excel(excel_path)

        else:

            nba_college_merge_df = self.merge_predictor_college()


        # Get normalized features based on strength of schedule
        nba_college_norm_df = self.norm_features(nba_college_merge_df)

        # Impute values of players which are to high - 3%
        nba_college_norm_df = self.outlier_corrector(nba_college_merge_df)

        # Drop variables with more than x% missing variables
        perc = self.null_column__drop * 100
        min_count =  int(((100 - perc) / 100) * nba_college_norm_df.shape[0] + 1)
        nba_college_norm_df = nba_college_norm_df.dropna( axis=1, thresh=min_count)

        self.save_feature_set(nba_college_norm_df, self.output_featureset_excel_preprocessed)

        return nba_college_norm_df

    def save_feature_set(self, df, feature_set_name):

        excel_path = os.path.join(self.top_path, self.data_folder, self.featureset_folder, feature_set_name)

        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 
        df.to_excel(writer, index=False)
        writer.save()

    def get_correlation_plots(self):

        excel_path = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_excel_preprocessed)

        if os.path.exists(excel_path):

            # Create folder if it doesnt exist
            plots_directory = os.path.join(self.top_path, self.data_folder, self.correlation_folder, self.success_var_folder)
            if not os.path.exists(excel_path):
            
                os.mkdir(plots_directory)

            # Delete previous plots in the folder
            for plot in os.listdir(plots_directory):
                os.remove(os.path.join(plots_directory, plot))

            # Load excel
            featureset_df = pd.read_excel(excel_path)

            # Drop variables not useful for correlation
            featureset_df.drop(columns=self.correlation_drop_vars, inplace=True)

            n_columns = len(featureset_df.columns)
            splits = int(np.ceil(n_columns / self.max_vars_correlation_plot))

            success_var = '{success_var}_{agg}'.format(success_var = self.nba_success_var, agg = self.success_var_agg)
            output_var_column_number = featureset_df.columns.get_loc(success_var)

            for split in range(splits):
                column_list = []

                for column in range(split * self.max_vars_correlation_plot, split * self.max_vars_correlation_plot + self.max_vars_correlation_plot):
                    if column < n_columns:
                        column_list.append(column)

                # Ensure the success measurement variable is in the plot, and if so, make sure is the last one
                if output_var_column_number not in column_list:
                    column_list.append(output_var_column_number)

                else:
                    column_list.remove(output_var_column_number)
                    column_list.append(output_var_column_number)

                
                split_df =  featureset_df.iloc[:, column_list]
                
                # Get seaborn plot 1
                plt.figure(figsize=(16, 6))
                heatmap = sns.heatmap(split_df.corr(),  cmap = 'coolwarm', vmin=-1, vmax=1, annot=True)

                heatmap.set_title(self.plot_1_title, fontdict={'fontsize':12}, pad=12)

                output_file_name = self.plot_1_file.format(var = self.nba_success_var,  number = split)
                path_to_output_plot = os.path.join(self.top_path, self.data_folder, self.correlation_folder, self.success_var_folder, output_file_name)
                plt.savefig(path_to_output_plot, bbox_inches='tight')

                # Get seaborn plot 2
                
                plt.figure(figsize=(8, 12))
                heatmap = sns.heatmap(split_df.corr()[[success_var]].sort_values(by=success_var, ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
                heatmap.set_title(self.plot_1_title, fontdict={'fontsize':18}, pad=16)
                output_file_name = self.plot_2_file.format(var = self.nba_success_var,  number = split)
                path_to_output_plot = os.path.join(self.top_path, self.data_folder, self.correlation_folder, self.success_var_folder, output_file_name)
                plt.savefig(path_to_output_plot, bbox_inches='tight')

            # Plot graphs with high correlation
            columns_high_corr = []
            featureset_corr = featureset_df.corr()

            for index, row in featureset_corr.iterrows():

                if (abs(row[success_var]) > self.high_correlation_threshold) & (success_var != index):
                    columns_high_corr.append([index, row[success_var]])

            for high_corr_var, corr in columns_high_corr:

                output_file_name = self.plot_3_file.format(x = high_corr_var,  y = success_var)
                path_to_output_plot = os.path.join(self.top_path, self.data_folder, self.correlation_folder, self.success_var_folder, output_file_name)

                plt.figure()
                plt.plot(featureset_df[high_corr_var], featureset_df[success_var], linestyle = 'none', marker = 'p')
                plt.xlabel(high_corr_var)
                plt.ylabel(success_var)
                plt.title(self.plot_3_title.format(corr = str(round(corr, 2)) , x = high_corr_var.upper(), y = success_var.upper()))
                plt.savefig(path_to_output_plot, bbox_inches='tight')

        else:

            print('Featureset {featureset} does not exist, build it first!'.format(featureset = self.output_featureset_excel))

