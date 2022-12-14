import pandas as pd
import os
from datetime import date
from os.path import dirname, abspath
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import random 
import csv
from sklearn.ensemble import RandomForestRegressor
import math
import numpy as np
import time

class DraftForecast:

  def __init__(self, config_variables):

    self.college_year_start = config_variables['SCRAPING_VARS']['COLLEGE']['START_YEAR']
    self.college_year_end = config_variables['SCRAPING_VARS']['COLLEGE']['END_YEAR']
    self.nba_year_start = config_variables['SCRAPING_VARS']['NBA']['START_YEAR']
    self.nba_year_end = config_variables['SCRAPING_VARS']['NBA']['END_YEAR']
    
    self.top_path = dirname(dirname(dirname(dirname(abspath(__file__)))))
    self.data_folder = config_variables['FOLDERS']['DATA']
    self.featureset_folder = config_variables['FOLDERS']['FEATURESET']
    self.hyperparameter_folder = config_variables['FOLDERS']['HYPERPARAMETER_TUNING']
    self.predictions_folder = config_variables['FOLDERS']['PREDICTIONS_FOLDER']
    self.predictions_plots_folder = config_variables['MODELING_PIPELINE']['PREDICTION']['FOLDER']['PLOTS']
    self.predictions_excel_folder = config_variables['MODELING_PIPELINE']['PREDICTION']['FOLDER']['EXCEL']
    self.feature_importance_folder = config_variables['MODELING_PIPELINE']['PREDICTION']['FOLDER']['IMPORTANCE_FOLDER']

    self.nba_success_var = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['NBA']['NBA_SUCCESS_VAR']
    self.success_var_agg = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['NBA']['SUCCESS_AGG']
    self.year_build_start = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['YEAR_BUILD']
    self.output_featureset_excel_preprocessed = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['FEATURESET_EXCEL_PREPROCESSED'].format(output = self.nba_success_var, agg = self.success_var_agg, build_year = self.year_build_start) 
    self.hyperparameter_var_folder = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['FOLDER'].format(var = self.nba_success_var, agg = self.success_var_agg)
    self.cv_folds = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['CV_FOLDS']

    self.train_test_split = config_variables['MODELING_PIPELINE']['TRAINING']['FEATURESET']['TEST_SPLIT']
    self.random_seed = config_variables['MODELING_PIPELINE']['RANDOM_SEED']
    self.features_list_drop = config_variables['MODELING_PIPELINE']['TRAINING']['FEATURESET']['COLUMNS_TO_DROP']

    self.n_hypertuning_iterations = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['ITERATIONS']
    self.n_trees = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['N_TREES']
    self.max_depth_setting = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['MAX_DEPTH_SETTING']
    self.max_depth_values = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['MAX_DEPTH_VALUES']
    self.max_features = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['MAX_FEATURES']
    self.hyperparameter_columns = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['HYPERPARAMETER_COLUMNS']
    self.evaluation_metric_model = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['HYPERPARAMETER_EVALUATION_MEASUREMENT']
    self.hyperparameter_excel = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['HYPERPARAMETER_STORING_FILE'].format(var = self.nba_success_var, agg = self.success_var_agg, n_fold = self.cv_folds, error = self.evaluation_metric_model)

    self.error_fold = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['N_FOLD_EVALUATION']
    self.error_fold_mean = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['ERROR_FOLD_MEAN']
    self.error_fold_std = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['ERROR_FOLD_STD']

    self.default_model_hyperparameters = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['MODEL_DEFAULT_PARAMETERS']
    self.comparison_to_default_metric = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['COMPARISON_TO_DEFAULT_MEASUREMENT']
    self.rookie_contract_length = config_variables['DATA_PREPROCESS_PIPELINE']['FEATURE_SET']['NBA']['ROOKIE_CONTRACT']
    
    self.draft_class_to_predict = config_variables['MODELING_PIPELINE']['PREDICTION']['PREDICTION_CLASS']
    self.prediction_error = config_variables['MODELING_PIPELINE']['PREDICTION']['EVALUATION_METRIC']
    self.draft_class_evaluation_columns = config_variables['MODELING_PIPELINE']['PREDICTION']['RELEVANT_COLUMNS_DRAFT_EVALUATION']
    self.success_var = '{var}_{agg}'.format(var = self.nba_success_var, agg = self.success_var_agg)
    self.draft_class_predictions_excel = config_variables['MODELING_PIPELINE']['PREDICTION']['DRAFT_CLASS_PREDICTIONS_EXCEL'].format(year = self.draft_class_to_predict, var = self.success_var)
    self.feature_importance_excel = config_variables['MODELING_PIPELINE']['PREDICTION']['FEATURE_IMPORTANCE_EXCEL'].format(year = self.draft_class_to_predict, var = self.success_var)
    self.draft_class_predictions_plot_1 = config_variables['MODELING_PIPELINE']['PREDICTION']['DRAFT_CLASS_PREDICTIONS_PLOT_1'].format(year = self.draft_class_to_predict, var = self.success_var)
    self.draft_class_predictions_plot_2 = config_variables['MODELING_PIPELINE']['PREDICTION']['DRAFT_CLASS_PREDICTIONS_PLOT_2'].format(year = self.draft_class_to_predict)
    self.current_year = date.today().year 

    self.hyperparameter_summary_plot_title_mean = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['HYPERPARAMETER_PLOT_TITLE_MEAN']
    self.hyperparameter_summary_plot_title_std = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['HYPERPARAMETER_PLOT_TITLE_STD']
    self.hyperparameter_summary_plot_file_mean = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['HYPERPARAMETER_PLOT_FILE_MEAN']
    self.hyperparameter_summary_plot_file_std = config_variables['MODELING_PIPELINE']['TRAINING']['HYPERPARAMETER_TUNING']['HYPERPARAMETER_PLOT_FILE_STD']

    self.feature_importance_plot_title = config_variables['MODELING_PIPELINE']['PREDICTION']['FEATURE_IMPORTANCE_PLOT_TITLE']
    self.feature_importance_plot_file = config_variables['MODELING_PIPELINE']['PREDICTION']['FEATURE_IMPORTANCE_PLOT_FILE']
    self.feature_importance_max_box = config_variables['MODELING_PIPELINE']['PREDICTION']['FEATURE_IMPORTANCE_MAX_BOX'] 

  def model_split_cv(self, random_seed):

    """
      Description:
          This function splits the feature set into training and test sets for the cross validation process

      Input Parameters:
          - config.yaml
          - random_seed: int - random seed to ensure results can be reproduced

      Return:
          - train_features
          - test_features
          - train_labels
          - test_labels
    """

    # Load df
    path_to_featureset_excel = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_excel_preprocessed)
    featureset = pd.read_excel(path_to_featureset_excel)

    # Drop players with less than 4 years in the nba - not fair training yet
    featureset = featureset[featureset['draft_year'] <= self.current_year - self.rookie_contract_length].reset_index(drop = True).copy()

    # Divide featureset into features and labels
    features = featureset.drop(columns=self.features_list_drop).copy()
    features = features.drop(columns=[self.success_var]).copy()
    labels = featureset[self.success_var].copy()

    # Split train_set and test_set
    train_features, test_features, train_labels, test_labels = sk.model_selection.train_test_split(features, labels,
                                                                                                test_size = self.train_test_split,
                                                                                                random_state = random_seed)

    return train_features, test_features, train_labels, test_labels

  def hyperparameter_df_generator(self):

    """
      Description:
          This function generates the dataframe where the hyperparameters will be stored

      Input Parameters:
          - config.yaml

      Return:
          - hyper_parameter_df: dataframe where the hyper parameters will be stored
    """

    fieldnames = []

    for column in self.hyperparameter_columns:
      fieldnames.append(column) 

    for fold in range(self.cv_folds):
      fieldnames.append(self.error_fold.format(n = str(fold + 1), evaluation = self.evaluation_metric_model))

    fieldnames.append(self.error_fold_mean)
    fieldnames.append(self.error_fold_std)

    hyper_parameter_df = pd.DataFrame(columns = fieldnames)

    return hyper_parameter_df

  def fit_model(self, n_trees, max_depth, max_features, train_features, train_labels):

    """
      Description:
          This function configurates the model (hyperparameters) and train the model

      Input Parameters:
          - config.yaml
          - n_trees: number of trees of the Random Forest.
          - max_depth: maximum depth of the trees.
          - max_features: maximum number of features used per tree.
          - train_features
          - train_labels

      Return:
          - rf: random forest model (trained)
    """

    rf = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, max_features = max_features, random_state = self.random_seed)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    return rf

  def calculate_metric_error(self, predictions, actuals, error_meeasurement):

    """
      Description:
          This function calculates the error between the prediction and the test labels (actuals).

      Input Parameters:
          - config.yaml
          - predictions: list containing the predictions from the random forest.
          - actuals: list containing the actual values (test labels).
          - error_meeasurement: string which specifies which error needs to be calculated.

      Return:
          - error_metric: float - the value of the error
    """

    if error_meeasurement == 'rmse':

      mse = np.square(np.subtract(actuals.to_numpy(),predictions)).mean()
      error_metric = math.sqrt(mse)

    elif error_meeasurement == 'mae':

      error_metric = abs(np.subtract(actuals.to_numpy(),predictions)).mean()

    elif error_meeasurement == 'mape':
      
      diff = np.subtract(actuals.to_numpy(),predictions)
      abs_perc_error = [abs(diff[row]/ actuals.to_numpy()[row]) if abs(actuals.to_numpy()[row]) >= 0.1 else abs(diff[row]/ 0.1) for row in range(len(diff))]
      error_metric = np.mean(abs_perc_error) * 100

    return error_metric
    
  def cross_validation(self, n_trees, max_depth, max_features, evaluation_metric):

    """
      Description:
          This function performs the cross validation of the model.

      Input Parameters:
          - config.yaml
          - n_trees: the number of trees of the random forest.
          - max_depth: the maximum depth of the tree.
          - max_features: the maximum features used per tree.
          - evaluation_metric: the error metric to be evaluated at the cross validation.

      Return:
          - error_results_dict: dictionary - contains the data of the cross validation (RM model parameters, each fold error,
                                              mean error across the folds and standandard deviation across the folds)
    """

    evaluation_metric_fold_array = []

    error_results_dict = {}
    error_results_dict['n_trees'] = n_trees
    error_results_dict['max_depth'] = max_depth
    error_results_dict['max_features'] = max_features

    for fold in range(self.cv_folds):

      # Split train_set and test_set
      train_features, test_features, train_labels, test_labels = self.model_split_cv(fold)

      # Fit the model
      rf = self.fit_model(n_trees, max_depth, max_features, train_features, train_labels)

      # Run Prediction
      rf_predictions = rf.predict(test_features)

      # Calculate metric error
      fold_error = self.calculate_metric_error(rf_predictions, test_labels, evaluation_metric)
      evaluation_metric_fold_array.append(fold_error)

      error_results_dict[self.error_fold.format(n = str(fold + 1), evaluation = evaluation_metric)] = fold_error

    # Get mean for k fold and std
    error_results_dict[self.error_fold_mean.format(evaluation = evaluation_metric)] = sum(evaluation_metric_fold_array) / len(evaluation_metric_fold_array)
    error_results_dict[self.error_fold_std.format(evaluation = evaluation_metric)] = np.std(evaluation_metric_fold_array)

    return error_results_dict

  def hyper_parameter_tuning(self):

    """
      Description:
          This function performs the hyperparameter tuning process

      Input Parameters:
          - config.yaml

      Return:
          - Saves into an excel all the hyperparameter iterations results.
    """

    start_time = time.time()

    # Check if folder to store results exists
    hyperparameter_tuning_path = os.path.join(self.top_path, self.data_folder, self.hyperparameter_folder, self.hyperparameter_var_folder)
    if not os.path.exists(hyperparameter_tuning_path):
      os.mkdir(hyperparameter_tuning_path)

    # Create dataframe to store hyperparameters
    hyper_parameter_df = self.hyperparameter_df_generator()

    for iteration in range(self.n_hypertuning_iterations):

      # Get random hyperparameters
      n_trees = random.randint(self.n_trees[0], self.n_trees[1])
      max_depth = random.choice([self.max_depth_setting[0], self.max_depth_setting[1]])
      if max_depth == self.max_depth_setting[0]:
        max_depth = random.randint(self.max_depth_values[0], self.max_depth_values[1])
      max_features = random.choice(self.max_features)

      # Perform Cross Validation and save results
      cv_results = self.cross_validation(n_trees, max_depth, max_features, self.evaluation_metric_model)
      # Get CV Results to hyperparameter tuning csv file
      hyper_parameter_df = hyper_parameter_df.append(cv_results, ignore_index=True)
      
    excel_path = os.path.join(hyperparameter_tuning_path, self.hyperparameter_excel)
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter') 
    hyper_parameter_df.to_excel(writer, index=False)
    writer.save()

    #Print Run time
    print("{step} took {time} minutes".format(step = 'The hyeperparameter tunning step', time = round((time.time() - start_time) / 60, 2)))

  def hypertuned_to_default_comparison(self, hypertuned_df):

    """
      Description:
          This function compares the performance of the best model found through hyperparameter tuning
          and the default Random Forest.

      Input Parameters:
          - config.yaml
          - hypertuned_df: the dataframe containing all the hyperparameter tunin iterations results.

      Return:
          - Prints on the console metrics comparing the performance of the hypertuned vs default Random Forests
    """

    default_trees = self.default_model_hyperparameters[0]
    default_depth = self.default_model_hyperparameters[1]
    default_features = self.default_model_hyperparameters[2]
    cv_error_default = self.cross_validation(default_trees, default_depth, default_features, self.comparison_to_default_metric)
    cv_error_mean_default = cv_error_default[self.error_fold_mean.format(evaluation = self.comparison_to_default_metric)]
    cv_error_std_default = cv_error_default[self.error_fold_std.format(evaluation = self.comparison_to_default_metric)]

    print('The default {error} is {number}'.format(error = self.comparison_to_default_metric, number = round(cv_error_mean_default, 3)))
    print('The default std {error} is {number}'.format(error = self.comparison_to_default_metric, number = round(cv_error_std_default, 4)))

    tuned_trees = hypertuned_df['n_trees'][0]
    tuned_depth = hypertuned_df['max_depth'][0]
    tuned_features = hypertuned_df['max_features'][0]
    cv_error_tuned = self.cross_validation(tuned_trees, tuned_depth, tuned_features, self.comparison_to_default_metric)
    cv_error_mean_tuned = cv_error_tuned[self.error_fold_mean.format(evaluation = self.comparison_to_default_metric)]
    cv_error_std_tuned = cv_error_tuned[self.error_fold_std.format(evaluation = self.comparison_to_default_metric)]

    print('The tuned mean {error} is {number}'.format(error = self.comparison_to_default_metric, number = round(cv_error_mean_tuned, 3)))
    print('The tuned std {error} is {number}'.format(error = self.comparison_to_default_metric, number = round(cv_error_std_tuned, 4)))

    cv_error_mean_improvement_abs = round(cv_error_mean_tuned - cv_error_mean_default, 3)
    cv_error_mean_improvemenet_perc = round(1 - (cv_error_mean_tuned - cv_error_mean_default) / cv_error_mean_default, 3)

    print('The hyperparameter tuning ({trees}, {depth}, {features}) reduced the mean {error} in predicting {var}_{agg} in {perc}% ({abs})'.format(trees = tuned_trees, depth = tuned_depth, features = tuned_features, error = self.comparison_to_default_metric,var = self.nba_success_var, agg = self.success_var_agg, perc = cv_error_mean_improvemenet_perc, abs = cv_error_mean_improvement_abs))

    cv_error_std_improvement_abs = round(cv_error_std_tuned - cv_error_std_default, 3)
    cv_error_std_improvemenet_perc = round(1 - (cv_error_std_tuned - cv_error_std_default) / cv_error_std_default, 3)

    print('The hyperparameter tuning ({trees}, {depth}, {features}) reduced the std {error} in predicting {var}_{agg} in {perc}% ({abs})'.format(trees = tuned_trees, depth = tuned_depth, features = tuned_features, error = self.comparison_to_default_metric,var = self.nba_success_var, agg = self.success_var_agg, perc = cv_error_std_improvemenet_perc, abs = cv_error_std_improvement_abs))

  def best_hyperparameter(self):

    """
      Description:
          Function to find the best hyperparameters from the hyperparameter tuning process

      Input Parameters:
          - config.yaml

      Return:
          - n_trees: the number of trees of the random forest.
          - max_depth: the maximum depth of the tree.
          - max_features: the maximum features used per tree.
    """

    # Load hyperparameters
    path_to_hyperparameters = os.path.join(self.top_path, self.data_folder, self.hyperparameter_folder, self.hyperparameter_var_folder, self.hyperparameter_excel)
    hyperparameters_df = pd.read_excel(path_to_hyperparameters)
    hyperparameters_df = hyperparameters_df.fillna(0)

    # Find top 10 and choose best trade off between metric mean and std
    top_hyperparameters_df = hyperparameters_df.sort_values(self.error_fold_mean.format(evaluation = self.evaluation_metric_model)).head(10)
    top_hyperparameters_df['diff_to_best_mean'] = top_hyperparameters_df[self.error_fold_mean.format(evaluation = self.evaluation_metric_model)] - top_hyperparameters_df[self.error_fold_mean.format(evaluation = self.evaluation_metric_model)].min()
    top_hyperparameters_df['diff_to_best_std'] = top_hyperparameters_df[self.error_fold_std.format(evaluation = self.evaluation_metric_model)] - top_hyperparameters_df[self.error_fold_std.format(evaluation = self.evaluation_metric_model)].min()
    top_hyperparameters_df['ranking'] = top_hyperparameters_df['diff_to_best_mean'] + top_hyperparameters_df['diff_to_best_std']
    top_hyperparameters_df['max_depth'] = [None if max_depth == 0 else int(max_depth) for max_depth in top_hyperparameters_df['max_depth']]
    top_hyperparameters_df = top_hyperparameters_df.sort_values('ranking').head(1).reset_index(drop = True).copy()

    # Calculate how much better the predictions from hypertuned are to default
    self.hypertuned_to_default_comparison(top_hyperparameters_df)

    n_trees = top_hyperparameters_df['n_trees'][0]
    max_depth = top_hyperparameters_df['max_depth'][0]
    max_features = top_hyperparameters_df['max_features'][0]

    return n_trees, max_depth, max_features

  def model_split_draft_class(self):

    """
      Description:
          Function to splits the featureset into train and predict sets based on the Draft class to predict

      Input Parameters:
          - config.yaml

      Return:
          - train_features
          - test_features
          - train_labels
          - test_labels
          - prediction_names: names of the players which will be predicted.
          - feature_list: list with the features names which were used in training.
    """

    # Load df
    path_to_featureset_excel = os.path.join(self.top_path, self.data_folder, self.featureset_folder, self.output_featureset_excel_preprocessed)
    featureset = pd.read_excel(path_to_featureset_excel)

    # Split train_set and test_set
    prediction_class = self.draft_class_to_predict
    features_to_drop = self.features_list_drop
    features_to_drop.append(self.success_var)
    train_features = featureset[(featureset['draft_year'] < prediction_class) &
                                (featureset['draft_year'] <= self.current_year - self.rookie_contract_length)].drop(columns=features_to_drop).copy()
    test_features = featureset[featureset['draft_year'] == prediction_class].drop(columns=features_to_drop).copy()
    train_labels = featureset[(featureset['draft_year'] < prediction_class) &
                                (featureset['draft_year'] <= self.current_year - self.rookie_contract_length)][self.success_var].copy()
    test_labels = featureset[featureset['draft_year'] == prediction_class][self.success_var].copy()
    prediction_names = featureset[featureset['draft_year'] == prediction_class].copy()

    feature_list = train_features.columns.to_list()

    return train_features, test_features, train_labels, test_labels, prediction_names, feature_list

  def prediction_draft_order(self, draft_player_names, actuals, predictions):

    """
      Description:
          Function to save the draft predictions into an excel and get visualizations of the predictions

      Input Parameters:
          - config.yaml
          - draft_player_names: the names of the players which are predicted.
          - actuals: the predict labels.
          - predictions: the predictions from the random forest.

      Return:
          - Saves a excel with the Draft prediction results and several plots/visualizations.
    """

    start_time = time.time()

    pred_error = self.calculate_metric_error(predictions, actuals, self.prediction_error)

    # Merge predictions to names
    draft_player_names['rf_prediction'] = predictions
    evaluation_draft_columns = self.draft_class_evaluation_columns
    evaluation_draft_columns.append(self.success_var)
    evaluation_draft_columns.append('rf_prediction')
    draft_player_names = draft_player_names[self.draft_class_evaluation_columns].sort_values('draft_overall').reset_index(drop = True).copy()
    draft_player_names = draft_player_names[self.draft_class_evaluation_columns].sort_values('rf_prediction', ascending = False).reset_index().copy()
    draft_player_names['draft_actual'] = draft_player_names['index'] + 1
    draft_player_names.drop(columns = 'index', inplace = True)
    draft_player_names['draft_prediction_ws'] = draft_player_names.index + 1
    success_var_order_list = sorted(draft_player_names[self.success_var].to_list(), reverse = True)
    draft_player_names['draft_actual_ws'] = [success_var_order_list.index(value) + 1 for value in draft_player_names[self.success_var]]
    draft_player_names['prediction_diff_to_actual'] = draft_player_names['draft_actual_ws'] - draft_player_names['draft_prediction_ws']
    draft_player_names['nba_gms_diff_to_actual'] = draft_player_names['draft_actual_ws'] - draft_player_names['draft_actual']

    
    # Print errors ws_mean
    mse = np.square(np.subtract(draft_player_names['ws_mean'], draft_player_names['rf_prediction'])).mean()
    rmse = math.sqrt(mse)
    print('The RMSE of the predictions is {rmse}'.format(rmse = rmse))

    mae = abs(np.subtract(draft_player_names['ws_mean'], draft_player_names['rf_prediction'])).mean()
    print('The MAE of the predictions is {mae}'.format(mae = mae))
      
    diff = np.subtract(draft_player_names['ws_mean'], draft_player_names['rf_prediction'])
    abs_perc_error = [abs(diff[row]/ actuals.to_numpy()[row]) if abs(actuals.to_numpy()[row]) >= 0.1 else abs(diff[row]/ 0.1) for row in range(len(diff))]
    mape = np.mean(abs_perc_error) * 100
    print('The MAPE of the predictions is {mape}'.format(mape = mape))


    # Save predictions
    output_directory = os.path.join(self.top_path, self.data_folder, self.predictions_folder)
    # Save excel
    path_to_excel = os.path.join(output_directory, self.predictions_excel_folder, self.draft_class_predictions_excel)
    writer = pd.ExcelWriter(path_to_excel, engine='xlsxwriter') 
    draft_player_names.to_excel(writer, index=False)
    writer.save()
    # Save figure - part 1
    path_to_plot = os.path.join(output_directory, self.predictions_plots_folder, self.draft_class_predictions_plot_1)
    draft_player_names = draft_player_names.sort_values(self.success_var, ascending = False).reset_index(drop = True).copy()
    plt.figure()
    plt.plot(draft_player_names.index, draft_player_names['rf_prediction'], label = 'model prediction')
    plt.plot(draft_player_names.index, draft_player_names[self.success_var], label = 'actual')
    plt.xlabel('Players sorted by Actual Win Shares')
    plt.ylabel(self.success_var)
    plt.legend()
    plt.title('Draft Class {year} - {var} prediction vs actuals - Mean {error} = {numeric}'.format(year = self.draft_class_to_predict, var = self.success_var, error = self.prediction_error, numeric = str(round(pred_error, 2))))
    plt.savefig(path_to_plot, bbox_inches='tight')

    # Save figure - part 2
    path_to_plot = os.path.join(output_directory, self.predictions_plots_folder, self.draft_class_predictions_plot_2)
    draft_player_names = draft_player_names.sort_values('draft_prediction_ws').copy()
    plt.figure()
    plt.plot(draft_player_names['draft_prediction_ws'], draft_player_names[self.success_var], label = 'Predictions Draft Order')
    draft_player_names = draft_player_names.sort_values('draft_actual').copy()
    plt.plot(draft_player_names['draft_actual'], draft_player_names[self.success_var], label = 'NBA Teams Draft Order (Actual)')
    plt.xlabel('Draft Order')
    plt.ylabel(self.success_var)
    plt.legend()
    plt.title('Draft Class {year} - NBA Picks WS vs Model Picks WS'.format(year = self.draft_class_to_predict))
    plt.savefig(path_to_plot, bbox_inches='tight')

    #Print Run time
    print("{step} took {time} minutes".format(step = 'The Model Evaluation step', time = round((time.time() - start_time) / 60, 2)))    

  def feature_importance(self, random_forest, feature_list):

    """
      Description:
          Function to analyzes the feature importance of the generated Random forest

      Input Parameters:
          - config.yaml
          - random_forest: random forest model (trained).
          - feature_list: a list with the name of all the features used to train the model.

      Return:
          - Saves a excel with the feature importance data from the Random forest and sevaral plots to visualize the feature importance.
    """

    # Get Importances
    importances = (random_forest.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Convert to dataframe and save to excel
    feature_importance_df = pd.DataFrame(columns = ['Feature', 'Feature Importance'])
    for data, importance in feature_importances:
      new_row = {'Feature': data, 'Feature Importance': importance}
      feature_importance_df = feature_importance_df.append(new_row, ignore_index=True)

    path_to_excel = os.path.join(self.top_path, self.data_folder, self.predictions_folder, self.feature_importance_folder, self.feature_importance_excel)
    feature_importance_df.to_excel(path_to_excel, index=False)

    # Create feature importance bar plot
    feature_importance_df_most_important = feature_importance_df.sort_values('Feature Importance', ascending=False).T.copy()
    splits = int(np.ceil(len(feature_importance_df_most_important.columns) / self.feature_importance_max_box))

    for split in range(splits):

      featureset_plot_list = feature_importance_df_most_important.columns.to_list()[split * self.feature_importance_max_box: split * self.feature_importance_max_box +  self.feature_importance_max_box]
      features_plot = feature_importance_df_most_important[featureset_plot_list].copy()

      plt.figure(figsize = (10, 5))
      plt.bar(features_plot.T['Feature'], features_plot.T['Feature Importance'], color = 'blue', width = 0.4)
      plt.xlabel('Feature')
      plt.ylabel('Feature Importance')
      plt.title(self.feature_importance_plot_title)
      plt.savefig(os.path.join(self.top_path, self.data_folder, self.predictions_folder, self.feature_importance_folder, self.feature_importance_plot_file.format(n = split)))

  def train_and_predict(self):

    """
      Description:
          Function to train the model and to make the predictions of the NBA class

      Input Parameters:
          - config.yaml

      Return:
          - It does not return any object, it calls other functions.
    """

    # Load hyperparameters and return best hyperparameter
    path_to_hyperparameters = os.path.join(self.top_path, self.data_folder, self.hyperparameter_folder, self.hyperparameter_var_folder, self.hyperparameter_excel)
    if os.path.exists(path_to_hyperparameters):
      n_trees, max_depth, max_features = self.best_hyperparameter()
    
    else:
      n_trees, max_depth, max_features = self.default_model_hyperparameters
      print('Need to perform hyperparameter tuning')

    # Get train sets and split sets based on class to be predicted
    start_time = time.time() 
    train_features, test_features, train_labels, test_labels, prediction_names, feature_list = self.model_split_draft_class()

    # Fit the model
    rf = self.fit_model(n_trees, max_depth, max_features, train_features, train_labels)
    #Print Run time
    print("{step} took {time} minutes".format(step = 'The Model Training step', time = round((time.time() - start_time) / 60, 2)))

    # Run Prediction
    start_time = time.time()
    rf_predictions = rf.predict(test_features)
    #Print Run time
    print("{step} took {time} minutes".format(step = 'The Model prediction step', time = round((time.time() - start_time) / 60, 2)))

    # Calculate metric error, plot, sort and save
    self.prediction_draft_order(prediction_names, test_labels, rf_predictions)

    # Get Feature Importance
    self.feature_importance(rf, feature_list)

  def hyperparameter_tuning_diagnosis(self):

    """
      Description:
          Function to analyze the results obtained from the hyper parameter tuning process.

      Input Parameters:
          - config.yaml

      Return:
          - Saves several plots to understand how the hyperparameters affect the performance of the Random Forest.
    """

    # Load hyperparameters and return best hyperparameter
    path_to_hyperparameters = os.path.join(self.top_path, self.data_folder, self.hyperparameter_folder, self.hyperparameter_var_folder, self.hyperparameter_excel)
    if os.path.exists(path_to_hyperparameters):

      #Print best parameters
      self.best_hyperparameter()

      #Plot RMSE and STD plots vs each variable
      path_to_hyperparameters_df = pd.read_excel(path_to_hyperparameters).fillna(0)

      for parameter in self.hyperparameter_columns:

        plt.figure()
        plt.plot(path_to_hyperparameters_df[parameter], path_to_hyperparameters_df[self.error_fold_mean.format(evaluation = self.evaluation_metric_model)], linestyle = 'none', marker = 'p')
        plt.xlabel(parameter)
        plt.ylabel(self.evaluation_metric_model)
        plt.title(self.hyperparameter_summary_plot_title_mean)
        plt.savefig(os.path.join(self.top_path, self.data_folder, self.hyperparameter_folder, self.hyperparameter_var_folder, self.hyperparameter_summary_plot_file_mean.format(parameter = parameter)))

        plt.figure()
        plt.plot(path_to_hyperparameters_df[parameter], path_to_hyperparameters_df[self.error_fold_std.format(evaluation = self.evaluation_metric_model)], linestyle = 'none', marker = 'p')
        plt.xlabel(parameter)
        plt.ylabel(self.evaluation_metric_model)
        plt.title(self.hyperparameter_summary_plot_title_std)
        plt.savefig(os.path.join(self.top_path, self.data_folder, self.hyperparameter_folder, self.hyperparameter_var_folder, self.hyperparameter_summary_plot_file_std.format(parameter = parameter)))

    else:

      print("Hyperparemeters file - {file} - does not exist". format(file = self.hyperparameter_excel))
