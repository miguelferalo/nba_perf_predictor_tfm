from statistics import linear_regression
import pandas as pd
import os
from datetime import date
from os.path import dirname, abspath
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time

class BasketDataProcessor(object):

    def __init__(self, config_variables):
        self.config = config_variables
        self.college_year_start = config_variables['SCRAPING_VARS']['COLLEGE']['START_YEAR']
        self.college_year_end = config_variables['SCRAPING_VARS']['COLLEGE']['END_YEAR']
        self.nba_year_start = config_variables['SCRAPING_VARS']['NBA']['START_YEAR']
        self.nba_year_end = config_variables['SCRAPING_VARS']['NBA']['END_YEAR']
        
        self.top_path = dirname(dirname(dirname(dirname(abspath(__file__)))))

        self.data_folder = config_variables['FOLDERS']['DATA']
        self.college_raw_folder = config_variables['FOLDERS']['COLLEGE_RAW']
        self.nba_raw_folder = config_variables['FOLDERS']['NBA_RAW']
        self.featureset_folder = config_variables['FOLDERS']['FEATURESET']
        self.correlation_folder = config_variables['FOLDERS']['CORRELATION']
        self.data_visualizations_folder = config_variables['FOLDERS']['DATA_VISUALIZATIONS']

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

        self.output_featureset_excel = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['FEATURESET_EXCEL'].format(output = self.nba_success_var, agg = self.success_var_agg, build_year = self.year_build_start)
        self.output_featureset_excel_preprocessed = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['FEATURESET_EXCEL_PREPROCESSED'].format(output = self.nba_success_var, agg = self.success_var_agg, build_year = self.year_build_start)  
        self.output_featureset_na_values_diagnosis = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['FEATURESET_NA_VALUES_DIAGNOSIS'].format(output = self.nba_success_var, agg = self.success_var_agg, build_year = self.year_build_start)      

        self.max_vars_correlation_plot = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['MAX_VARS_PLOT']
        self.success_var_folder = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['FOLDER']['SUCCESS_VAR'].format(var = self.nba_success_var.upper())
        self.correlation_drop_vars = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['NOT_RELEVANT_COLUMNS']
        self.high_correlation_threshold = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['HIGH_CORR_THRESHOLD']

        self.plot_1_title = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_1_TITLE'].format(var = self.nba_success_var.upper())
        self.plot_3_title = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_3_TITLE']
        self.plot_4_title = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_4_TITLE']
        self.plot_1_file = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_1_FILE']
        self.plot_2_file = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_2_FILE']  
        self.plot_3_file = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_3_FILE']  
        self.plot_4_file = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['PLOT_TYPE_4_FILE']          

        self.norm_var = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['NORM_VAR']
        self.vars_to_norm = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['NORMALIZED_FEATURES']
        self.outlier_vars = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['OUTLIER_DETECTOR_VARS']

        self.null_column_drop_threshold = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['NULL_COLUMNS_DROP']
        self.null_column_avg_imputation_threshold = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['COLLEGE']['NULL_COLUMNS_AVG_UMPUTATION']
        self.features_list_drop = config_variables['MODELING_PIPELINE']['TRAINING']['FEATURESET']['COLUMNS_TO_DROP']

        self.variables_to_impute = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['IMPUTATION']['VARS_TO_IMPUTE']
        self.linear_regression_vars = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['IMPUTATION']['REGRESSION_VAR']
        self.histogram_bins = config_variables['DATA_PREPROCESS_PIPELINE']['CORRELATION']['HISTOGRAM_BINS']

        self.features_analysis_breakdown_columns = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET_SUMMARY']['FEATURES_BREAKDOWN_COLUMNS']
        self.features_description = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET_SUMMARY']['FEATURES_DESCRIPTION'].format(output = self.nba_success_var, agg = self.success_var_agg, build_year = self.year_build_start)
        self.boxplot_title = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET_SUMMARY']['BOXPLOT_TITLE']
        self.boxplot_output_file = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET_SUMMARY']['BOXPLOT_FILE']
        self.max_boxes = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET_SUMMARY']['BOXPLOT_MAX_BOXES_GRAPH']

    def nba_df_filter(self):

        path_to_nba_excel = os.path.join(self.top_path, self.data_folder, self.nba_raw_folder, self.nba_raw_excel)

        # Load df
        nba_df = pd.read_excel(path_to_nba_excel, sheet_name = self.nba_advanced_table)

        nba_df_rookie = nba_df[((nba_df['season'] - nba_df['draft_year']) <= self.rookie_contract_length + 1) &
                                (nba_df['draft_year'] > self.year_build_start) & 
                                (nba_df['draft'] == 1)].reset_index(drop = True).copy()

        if self.success_var_agg == 'mean':
            
            nba_df_rookie = nba_df_rookie.groupby(['id', 'draft_overall', 'draft_year', 'name', 'heigth_cm'], as_index=False)[self.nba_success_var].mean()
            nba_df_rookie.columns = ['id', 'draft_overall', 'draft_year', 'name', 'heigth_cm', '{success_var}_{agg}'.format(success_var = self.nba_success_var, agg = self.success_var_agg)]
        else:
            nba_df_rookie = nba_df_rookie.groupby(['id', 'draft_overall', 'draft_year', 'name', 'heigth_cm'], as_index=False)[self.nba_success_var].sum()
            nba_df_rookie.columns = ['id', 'draft_overall', 'draft_year', 'name', 'heigth_cm', '{success_var}_{agg}'.format(success_var = self.nba_success_var, agg = self.success_var_agg)]

        return nba_df_rookie

    def college_specific_data_columns(self, college_last_df, college_df, columns_spec):

        # Drop duplicates
        college_last_df = college_last_df.drop_duplicates(subset=['id'], keep='last')

        for column in columns_spec:

            if 'n_seasons' in column:

                n_season = college_df.groupby(['id'], as_index=False)['season'].count()
                n_season.columns = ['id', 'n_seasons']

                college_last_df = college_last_df.merge(n_season, on=['id'],
                                                                how='left')  

            if 'position' in column:

                position = college_df.groupby(['id'], as_index=False)['position'].last()
                position.columns = ['id', 'position']

                college_last_df = college_last_df.merge(position, on=['id'],
                                                                how='left')

            if 'school' in column:

                school = college_df.groupby(['id'], as_index=False)['school'].last()
                school.columns = ['id', 'school']

                college_last_df = college_last_df.merge(school, on=['id'],
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
        college_df_last = college_df.groupby(['id','name'], as_index=False).last('season')

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

            college_per_40_m_df = self.college_df_loader(self.college_per_40_m_table)

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

    def missing_values_imputation(self, df):

        # Get a missing values dictionary
        missing_values_dict = {}

        df_na_checker = df.copy()
        df_na_checker = df_na_checker.drop(columns=self.features_list_drop)


        for column in df_na_checker.columns:

            number_missing = df_na_checker[column].isna().sum()
            total_values = len(df_na_checker)
            perc_missing_values = round(number_missing * 100/ total_values, 2)
            missing_values_dict[column] = perc_missing_values

        missing_values_df = pd.DataFrame([missing_values_dict])
        missing_values_df = missing_values_df.T.copy()
        missing_values_df['data column'] = missing_values_df.index
        missing_values_df.columns = ['missing_values %', 'data column']
        missing_values_df = missing_values_df[['data column', 'missing_values %']].sort_values('missing_values %', ascending=False)

        # Save missing values dataframe to excel
        self.save_feature_set(missing_values_df, self.output_featureset_na_values_diagnosis)

        # Drop columns with a high proportion of null values
        perc = self.null_column_drop_threshold * 100
        min_count =  int(((100 - perc) / 100) * df.shape[0] + 1)
        df = df.dropna(axis=1, thresh=min_count)

        # Apply linear regression imputation on the remaining varibales with missing na
        for index, na_column in enumerate(self.variables_to_impute):

            if na_column == '3p% (game)':
                df[na_column] = df[na_column].fillna(0)

                # Impute values of players which are to high - 3%
                df = self.outlier_corrector(df)

            else:

                # calculating value of coefficients in case of linear polynomial
                df_regression = df.dropna(axis=0, subset=[na_column]).reset_index(drop = True).copy()
                x = df_regression[self.linear_regression_vars[index]].to_list()
                y = df_regression[na_column].to_list()
                linear_regression = np.poly1d(np.polyfit(x, y, 1))

                df[na_column] = [linear_regression(df[self.linear_regression_vars[index]][row]) if math.isnan(value) else value for row, value in enumerate(df[na_column])]


        return df

    def build_featureset(self):

        excel_path = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_excel)

        if os.path.exists(excel_path):

            nba_college_merge_df = pd.read_excel(excel_path)

        else:

            start_time = time.time()

            nba_college_merge_df = self.merge_predictor_college()

            #Print Run time
            print("{step} took {time} minutes".format(step = 'The preprocessed featureset generation step', time = round((time.time() - start_time) / 60, 2)))

        start_time = time.time()

        # Get normalized features based on strength of schedule
        nba_college_norm_df = self.norm_features(nba_college_merge_df)

        # Tranform Position variable to numeric
        nba_college_norm_df['position_cat'] = nba_college_norm_df['position'].astype('category').cat.codes
        nba_college_norm_df['school_cat'] = nba_college_norm_df['school'].astype('category').cat.codes

        # Drop variables with more than x% missing variables, treat imputation
        nba_college_norm_df = self.missing_values_imputation(nba_college_norm_df)

        # Drop variables with two entries same player - two schools at last season
        nba_college_norm_df = nba_college_norm_df.groupby('id', as_index=False).last().copy()

        # Save Featureset to excel
        self.save_feature_set(nba_college_norm_df, self.output_featureset_excel_preprocessed)

        #Print Run time
        print("{step} took {time} minutes".format(step = 'The Feature set imputation step', time = round((time.time() - start_time) / 60, 2)))

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

    def data_visualizer(self):

        feature_set_excel_path = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_excel)
        na_values_excel_path = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_na_values_diagnosis)

        if os.path.exists(feature_set_excel_path) & os.path.exists(na_values_excel_path):

            # Load featureset and missing_values dataset
            featureset = pd.read_excel(feature_set_excel_path)
            featureset.drop(columns=self.correlation_drop_vars, inplace=True)
            missing_values_df = pd.read_excel(na_values_excel_path)
            # Get only missing column below minimum before dropping and abobe minimum before imputing with mean
            vars_no_missing_values = missing_values_df[missing_values_df['missing_values %'] == 0]['data column'].to_list()
            missing_values_df = missing_values_df[(missing_values_df['missing_values %'] <= self.null_column_drop_threshold * 100) &
                                                (missing_values_df['missing_values %'] > self.null_column_avg_imputation_threshold * 100)].copy()
            
            # Loop through missing value columns and get the variable with higher correlation
            missing_var_list = []
            corr_df = featureset.corr()
            for column in missing_values_df['data column']:

                for row in range(len(corr_df)):
                    highest_correlated_corr = corr_df.sort_values(column, ascending = False).reset_index(drop = True)[column][row]
                    if corr_df[corr_df[column] == highest_correlated_corr].index[0] in vars_no_missing_values:
                        highest_correlated_var = corr_df[corr_df[column] == highest_correlated_corr].index[0]
                        break
                missing_var_list.append([column, highest_correlated_var, highest_correlated_corr])

            # Plot Visualization on missing value column and highest correlated variable
            for na_column, highest_correlated_var, highest_correlated_corr in missing_var_list:

                output_file_name = self.plot_3_file.format(x = highest_correlated_var,  y = na_column)
                path_to_output_plot = os.path.join(self.top_path, self.data_folder, self.data_visualizations_folder, output_file_name)

                plt.figure()
                plt.plot(featureset[highest_correlated_var], featureset[na_column], linestyle = 'none', marker = 'p')
                plt.xlabel(highest_correlated_var)
                plt.ylabel(na_column)
                plt.title(self.plot_3_title.format(corr = str(round(highest_correlated_corr, 2)) , y = highest_correlated_var.upper(), x = na_column.upper()))
                plt.savefig(path_to_output_plot, bbox_inches='tight')

            # Plot Visualization of 3pt var
            output_file_name = '3pt%_closer_look.png'
            path_to_output_plot = os.path.join(self.top_path, self.data_folder, self.data_visualizations_folder, output_file_name)
            featureset_reduced = featureset[featureset['3p (game)'] < 1].copy()

            plt.figure()
            plt.plot(featureset_reduced['3p (game)'], featureset_reduced['3p% (game)'], linestyle = 'none', marker = 'p')
            plt.xlabel('3p (game)')
            plt.ylabel('3p% (game)')
            plt.title('Closer look at 3p% from players with low 3p')
            plt.savefig(path_to_output_plot, bbox_inches='tight')

            # Plot Visualization of success var
            success_var = '{success_var}_{agg}'.format(success_var = self.nba_success_var, agg = self.success_var_agg)
            output_file_name = self.plot_4_file.format(success_var = success_var)
            path_to_output_plot = os.path.join(self.top_path, self.data_folder, self.data_visualizations_folder, output_file_name)

            plt.figure()
            plt.hist(x=[featureset[success_var]], bins=self.histogram_bins)
            plt.xlabel(success_var)
            plt.ylabel('Number of Players')
            plt.title(self.plot_4_title.format(success_var = success_var.upper()))
            plt.savefig(path_to_output_plot, bbox_inches='tight')

        elif os.path.exists(feature_set_excel_path):

            print('The missing values excel does not exist')

        elif os.path.exists(na_values_excel_path):

            print('The featureset does not exists')

        else:

            print('Both Files do not exist')


    def definitive_featureset_summary(self):

        feature_set_def_excel_path = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_excel_preprocessed)

        if os.path.exists(feature_set_def_excel_path):

            # Get a dataframe with the distribution of the features
            featureset = pd.read_excel(feature_set_def_excel_path)
            featureset_analysis = featureset.drop(columns=self.college_merging_columns).copy()
            
            rows_list = []
            for column in featureset_analysis.columns:
                row_dict = {}
                row_dict[self.features_analysis_breakdown_columns[0]] = column
                row_dict[self.features_analysis_breakdown_columns[1]] = featureset_analysis[column].mean()
                row_dict[self.features_analysis_breakdown_columns[2]] = featureset_analysis[column].std()
                row_dict[self.features_analysis_breakdown_columns[3]] = featureset_analysis[column].min()
                row_dict[self.features_analysis_breakdown_columns[4]] = featureset_analysis[column].max()
                row_dict[self.features_analysis_breakdown_columns[5]] = featureset_analysis[column].quantile(0.25)
                row_dict[self.features_analysis_breakdown_columns[6]] = featureset_analysis[column].quantile(0.5)
                row_dict[self.features_analysis_breakdown_columns[7]] = featureset_analysis[column].quantile(0.75)
                
                rows_list.append(row_dict)

            features_breakdown = pd.DataFrame(rows_list)   
            path_to_output_folder = os.path.join(self.top_path, self.data_folder, self.featureset_folder, '{var}_{agg}'.format(var = self.nba_success_var, agg = self.success_var_agg))
            if not os.path.exists(path_to_output_folder):
            
                os.mkdir(path_to_output_folder)

            path_to_output_excel = os.path.join(path_to_output_folder, self.features_description)
            # Save to excel
            features_breakdown.to_excel(path_to_output_excel, index=False)

            # Create boxplots for the features
            n_columns = len(featureset_analysis.columns)
            splits = int(np.ceil(n_columns / self.max_boxes))

            sorted_columns_values = features_breakdown.sort_values(self.features_analysis_breakdown_columns[4])[self.features_analysis_breakdown_columns[0]].to_list()

            for split in range(splits):

                #Get splits columns
                split_columns = []
                for column in sorted_columns_values[split * self.max_boxes:split * self.max_boxes + self.max_boxes]:
                    split_columns.append(column)
                
                if len(split_columns) > 0:

                    reduced_df = featureset_analysis[split_columns].copy()
                    path_to_output_boxplot = os.path.join(path_to_output_folder, self.boxplot_output_file.format(n = split))
                    
                    plt.figure()
                    sns.boxplot(x="variable", y="value", data=pd.melt(reduced_df))
                    plt.title(self.boxplot_title)
                    plt.savefig(path_to_output_boxplot, bbox_inches='tight')


        else:

            print('The definitive featureset does not exists')



