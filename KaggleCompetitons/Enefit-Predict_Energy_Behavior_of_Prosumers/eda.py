import pandas as pd
import numpy as np
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from geopy.geocoders import Nominatim

import xgboost as xgb
import catboost as cat

def prepare_geoopt(data_dir:str,forecast_weather:pd.DataFrame)->pd.DataFrame:
    """
    Using the name of the county (from county_id_to_name_map.json)
    file, and 'geopy' library, compute the location (latitute, longitute)
    of the county.
    This is needed to join the weather file with train data.
    Function is inspired by:
    https://www.kaggle.com/code/fabiendaniel/mapping-locations-and-county-codes
    combining with
    https://www.kaggle.com/datasets/michaelo/fabiendaniels-mapping-locations-and-county-codes/data
    :return:
    """
    # files = list(Path("./").glob("*.csv"))
    # for file in files:
    #     print(f"creates: '{file.stem}'")
    #     globals()[file.stem] = pd.read_csv(file)
    f = open(data_dir+"/county_id_to_name_map.json")

    county_codes = json.load(f)
    print(f"Loaded following county codes {county_codes}")

    parsed_counties = {v.lower().rstrip("maa"): k for k, v in county_codes.items()}
    print(f"Parsed the following counties {parsed_counties}")

    name_mapping = {
        "valga": "valg",
        "põlva": "põlv",
        "jõgeva": "jõgev",
        "rapla": "rapl",
        "järva": "järv"
    }

    # forecast_weather = pd.read_csv("./data/"+"forecast_weather.csv")

    tmp_cords = forecast_weather[["latitude", "longitude"]].drop_duplicates()

    parsed_counties_clean = {name_mapping.get(k, k): v for k, v in parsed_counties.items()}
    county_data = {v: [] for _, v in parsed_counties_clean.items()}

    for i, coords in tmp_cords.iterrows():

        lat, lon = coords["latitude"], coords["longitude"]

        print(f"Processing {i}/{len(tmp_cords)} lat={lat} lon={lon}")

        geoLoc = Nominatim(user_agent="GetLoc")

        # passing the coordinates
        locname = geoLoc.reverse(f"{lat}, {lon}")  # lat, lon

        if locname is None:
            continue
        location = locname.raw["address"]
        # print(f"\tCountry is {location['country']} County is {location['county']}")


        if location["country"] == "Eesti":
            county = location['county'].split()[0].lower()
            county = name_mapping.get(county, county)
            print(f"\tEesti county: '{county}', county code:", parsed_counties[county], "Coordiantes: ", (lat, lon))
            county_data[parsed_counties_clean[county]].append((lat, lon))
        else:
            print(f"\tDifferent country: {location['country']}")

    df_data = {"county": [], "longitude": [], "latitude": []}
    for k, v in county_data.items():
        df_data["county"] += [k] * len(v)
        df_data["latitude"] += [x[0] for x in v]
        df_data["longitude"] += [x[1] for x in v]

    new_df = pd.DataFrame(df_data)
    new_df["county"] = df["county"].astype(object) # co compatibility with other data

    print(f"Saving dataframe with {new_df.shape} shape")

    new_df.to_csv(data_dir+"county_lon_lats.csv") # ,index=False

    return new_df

class FeatureProcessorClass():
    def __init__(self):
        # Columns to join on for the different datasets
        self.weather_join = ['datetime', 'county', 'data_block_id']
        self.gas_join = ['data_block_id']
        self.electricity_join = ['datetime', 'data_block_id']
        self.client_join = ['county', 'is_business', 'product_type', 'data_block_id']

        # Columns of latitude & longitude
        self.lat_lon_columns = ['latitude', 'longitude']

        # Aggregate stats
        self.agg_stats = ['mean']  # , 'min', 'max', 'std', 'median']

        # Categorical columns (specify for XGBoost)
        self.category_columns = ['county', 'is_business', 'product_type', 'is_consumption', 'data_block_id']

    def __call__(self,
                 data,
                 client,
                 historical_weather,
                 forecast_weather,
                 electricity,
                 gas,
                 location
                 ):
        '''Processing of features from all datasets, merge together and return features for dataframe df '''
        # Create features for relevant dataset
        data = self.create_data_features(data)
        client = self.create_client_features(client)
        historical_weather = self.create_historical_weather_features(historical_weather, location)
        forecast_weather = self.create_forecast_weather_features(forecast_weather, location)
        electricity = self.create_electricity_features(electricity)
        gas = self.create_gas_features(gas)

        # Merge all datasets into one df
        df = data.merge(client, how='left', on=self.client_join)
        df = df.merge(historical_weather, how='left', on=self.weather_join)
        df = df.merge(forecast_weather, how='left', on=self.weather_join)
        df = df.merge(electricity, how='left', on=self.electricity_join)
        df = df.merge(gas, how='left', on=self.gas_join)

        # Change columns to categorical for XGBoost
        df[self.category_columns] = df[self.category_columns].astype('category')
        return df

    def create_new_column_names(self, df, suffix, columns_no_change):
        '''Change column names by given suffix, keep columns_no_change, and return back the data'''
        df.columns = [col + suffix
                      if col not in columns_no_change
                      else col
                      for col in df.columns
                      ]
        return df

    def flatten_multi_index_columns(self, df):
        df.columns = ['_'.join([col for col in multi_col if len(col) > 0])
                      for multi_col in df.columns]
        return df

    def create_data_features(self, data):
        '''Create features for main data (test or train) set'''
        # To datetime
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Time period features
        data['date'] = data['datetime'].dt.normalize()
        data['year'] = data['datetime'].dt.year
        data['quarter'] = data['datetime'].dt.quarter
        data['month'] = data['datetime'].dt.month
        data['week'] = data['datetime'].dt.isocalendar().week
        data['hour'] = data['datetime'].dt.hour

        # Day features
        data['day_of_year'] = data['datetime'].dt.day_of_year
        data['day_of_month'] = data['datetime'].dt.day
        data['day_of_week'] = data['datetime'].dt.day_of_week
        return data

    def create_client_features(self, client):
        ''' Create client features '''
        # Modify column names - specify suffix
        client = self.create_new_column_names(client,
                                           suffix='_client',
                                           columns_no_change = self.client_join
                                          )
        return client

    def create_historical_weather_features(self, historical_weather, location):
        '''Create historical weather features'''

        # To datetime
        historical_weather['datetime'] = pd.to_datetime(historical_weather['datetime'])

        # Add county
        historical_weather[self.lat_lon_columns] = historical_weather[self.lat_lon_columns].astype(float).round(1)
        historical_weather = historical_weather.merge(location, how='left', on=self.lat_lon_columns)

        # Modify column names - specify suffix
        historical_weather = self.create_new_column_names(historical_weather,
                                                          suffix='_h',
                                                          columns_no_change=self.lat_lon_columns + self.weather_join
                                                          )

        # Group by & calculate aggregate stats
        agg_columns = [col for col in historical_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        historical_weather = historical_weather.groupby(self.weather_join).agg(agg_dict).reset_index()

        # Flatten the multi column aggregates
        historical_weather = self.flatten_multi_index_columns(historical_weather)

        # Test set has 1 day offset for hour<11 and 2 day offset for hour>11
        historical_weather['hour_h'] = historical_weather['datetime'].dt.hour
        historical_weather['datetime'] = (historical_weather
                                          .apply(lambda x:
                                                 x['datetime'] + pd.DateOffset(1)
                                                 if x['hour_h'] < 11
                                                 else x['datetime'] + pd.DateOffset(2),
                                                 axis=1)
                                          )
        return historical_weather

    def create_forecast_weather_features(self, forecast_weather, location):
        '''Create forecast weather features'''

        # Rename column and drop
        forecast_weather = (forecast_weather
                            .rename(columns={'forecast_datetime': 'datetime'})
                            .drop(columns='origin_datetime')  # not needed
                            )

        # To datetime
        forecast_weather['datetime'] = (pd.to_datetime(forecast_weather['datetime'])
                                        .dt
                                        .tz_convert('Europe/Brussels')  # change to different time zone?
                                        .dt
                                        .tz_localize(None)
                                        )

        # Add county
        forecast_weather[self.lat_lon_columns] = forecast_weather[self.lat_lon_columns].astype(float).round(1)
        forecast_weather = forecast_weather.merge(location, how='left', on=self.lat_lon_columns)

        # Modify column names - specify suffix
        forecast_weather = self.create_new_column_names(forecast_weather,
                                                        suffix='_f',
                                                        columns_no_change=self.lat_lon_columns + self.weather_join
                                                        )

        # Group by & calculate aggregate stats
        agg_columns = [col for col in forecast_weather.columns if col not in self.lat_lon_columns + self.weather_join]
        agg_dict = {agg_col: self.agg_stats for agg_col in agg_columns}
        forecast_weather = forecast_weather.groupby(self.weather_join).agg(agg_dict).reset_index()

        # Flatten the multi column aggregates
        forecast_weather = self.flatten_multi_index_columns(forecast_weather)
        return forecast_weather

    def create_electricity_features(self, electricity):
        '''⚡ Create electricity prices features ⚡'''
        # To datetime
        electricity['forecast_date'] = pd.to_datetime(electricity['forecast_date'])

        # Test set has 1 day offset
        electricity['datetime'] = electricity['forecast_date'] + pd.DateOffset(1)

        # Modify column names - specify suffix
        electricity = self.create_new_column_names(electricity,
                                                   suffix='_electricity',
                                                   columns_no_change=self.electricity_join
                                                   )
        return electricity

    def create_gas_features(self, gas):
        '''Create gas prices features'''
        # Mean gas price
        gas['mean_price_per_mwh'] = (gas['lowest_price_per_mwh'] + gas['highest_price_per_mwh']) / 2

        # Modify column names - specify suffix
        gas = self.create_new_column_names(gas,
                                           suffix='_gas',
                                           columns_no_change=self.gas_join
                                           )
        return gas


def create_revealed_targets_train(data, N_day_lags):
    '''Create past revealed_targets for train set based on number of day lags N_day_lags'''
    original_datetime = data['datetime']
    revealed_targets = data[['datetime', 'prediction_unit_id', 'is_consumption', 'target']].copy()

    # Create revealed targets for all day lags
    for day_lag in range(2, N_day_lags + 1):
        revealed_targets['datetime'] = original_datetime + pd.DateOffset(day_lag)
        data = data.merge(revealed_targets,
                          how='left',
                          on=['datetime', 'prediction_unit_id', 'is_consumption'],
                          suffixes=('', f'_{day_lag}_days_ago')
                          )
    return data

def preprocess_datasets(data_dir):
    # load all datasets
    train = pd.read_csv(data_dir + "train.csv")
    client = pd.read_csv(data_dir + "client.csv")
    historical_weather = pd.read_csv(data_dir + "historical_weather.csv")
    forecast_weather = pd.read_csv(data_dir + "forecast_weather.csv")
    electricity = pd.read_csv(data_dir + "electricity_prices.csv")
    gas = pd.read_csv(data_dir + "gas_prices.csv")

    if (IS_LOCAL):
        location = pd.read_csv(data_dir + "county_lon_lats.csv")
    else:
        location = prepare_geoopt(data_dir, forecast_weather)
    print("Data Loaded Successfully")

    # Create all features
    N_day_lags = 15  # Specify how many days we want to go back (at least 2)

    FeatureProcessor = FeatureProcessorClass()

    # collate all datasets together
    data = FeatureProcessor(data=train.copy(),
                            client=client.copy(),
                            historical_weather=historical_weather.copy(),
                            forecast_weather=forecast_weather.copy(),
                            electricity=electricity.copy(),
                            gas=gas.copy(),
                            location=location.copy()
                            )

    print("Data is collated successfully")

    # add time-lagged data to dataset
    df = create_revealed_targets_train(data.copy(),
                                       N_day_lags=N_day_lags)
    print("Lagged data is added successfully")

    # save dataset
    df.to_csv(data_dir + "collated_train.csv", index=False)
    print("Resulting dataframe is saved successfully")

def load_reduce_memory_dataset(data_dir)->pd.DataFrame:

    data = pd.read_csv(data_dir + "collated_train.csv")

    start_mem = data.memory_usage().sum() / 1024 ** 2
    print('Memory usage before optimization is: {:.2f} MB'.format(start_mem))

    for col in list(data):
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint8)
                    else:
                        data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint16)
                    else:
                        data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint64)
                    else:
                        data[col] = data[col].astype(np.int64)
            else:
                #             if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #                 data[col] = data[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    #data.to_csv(data_dir + "collated_reduced_train.csv", index=False)
    print("Memory reduced dataframe is saved successfully")
    return data



# --------------------------------------------------------

def model_xgboost(df: pd.DataFrame, features:list[str]):
    #### Create single fold split ######

    # df = df.select_dtypes(exclude=['object'])

    train_block_id = list(range(0, 600)) # out of 637

    tr = df[df['data_block_id'].isin(train_block_id)]  # first 600 data_block_ids used for training
    val = df[~df['data_block_id'].isin(train_block_id)]  # rest data_block_ids used for validation

    # GPU or CPU use for model
    # GPU or CPU use for model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    DEBUG = False  # False/True

    clf = xgb.XGBRegressor(
        device=device,
        enable_categorical=True,
        objective='reg:absoluteerror',
        n_estimators=100 if DEBUG else 1500,
        early_stopping_rounds=100
    )

    clf.fit(X=tr[features],
            y=tr[target],
            eval_set=[(tr[features], tr[target]),
                      (val[features], val[target])],
            verbose=True  # False #True
            )

    print(f'Early stopping on best iteration #{clf.best_iteration} '
          f'with MAE error on validation set of {clf.best_score:.2f}')

    # Plot RMSE
    results = clf.evals_result()
    train_mae, val_mae = results["validation_0"]["mae"], results["validation_1"]["mae"]
    x_values = range(0, len(train_mae))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_values, train_mae, label="Train MAE")
    ax.plot(x_values, val_mae, label="Validation MAE")
    ax.legend()
    plt.ylabel("MAE Loss")
    plt.title("XGBoost MAE Loss")
    plt.show()

    # plot
    TOP = 20
    importance_data = pd.DataFrame(
        {'name': clf.feature_names_in_,
         'importance': clf.feature_importances_}
    )
    importance_data = importance_data.sort_values(by='importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=importance_data[:TOP],
                x='importance',
                y='name'
                )
    patches = ax.patches
    count = 0
    for patch in patches:
        height = patch.get_height()
        width = patch.get_width()
        perc = 100 * importance_data['importance'].iloc[count]  # 100*width/len(importance_data)
        ax.text(width, patch.get_y() + height / 2, f'{perc:.1f}%')
        count += 1

    plt.title(f'The top {TOP} features sorted by importance')
    plt.show()

def create_revealed_targets_test(data, previous_revealed_targets, N_day_lags):
    '''Create new test data based on previous_revealed_targets and N_day_lags '''
    for count, revealed_targets in enumerate(previous_revealed_targets):
        day_lag = count + 2

        # Get hour
        revealed_targets['hour'] = pd.to_datetime(revealed_targets['datetime']).dt.hour

        # Select columns and rename target
        revealed_targets = revealed_targets[['hour', 'prediction_unit_id', 'is_consumption', 'target']]
        revealed_targets = revealed_targets.rename(columns={"target": f"target_{day_lag}_days_ago"})

        # Add past revealed targets
        data = pd.merge(data,
                        revealed_targets,
                        how='left',
                        on=['hour', 'prediction_unit_id', 'is_consumption'],
                        )

    # If revealed_target_columns not available, replace by nan
    all_revealed_columns = [f"target_{day_lag}_days_ago" for day_lag in range(2, N_day_lags + 1)]
    missing_columns = list(set(all_revealed_columns) - set(data.columns))
    data[missing_columns] = np.nan

    return data

def plotting(df:pd.DataFrame):
    pass


if __name__ == '__main__':

    df = load_reduce_memory_dataset(data_dir="./data/")
    plotting(df)


    # Settings
    DEBUG = False #
    IS_LOCAL = True #
    data_dir = "./data/"

    ''' ------------| PROCESS TRAIN DATA |------------ '''

    # load all datasets
    train = pd.read_csv(data_dir + "train.csv")
    client = pd.read_csv(data_dir + "client.csv")
    historical_weather = pd.read_csv(data_dir + "historical_weather.csv")
    forecast_weather = pd.read_csv(data_dir + "forecast_weather.csv")
    electricity = pd.read_csv(data_dir + "electricity_prices.csv")
    gas = pd.read_csv(data_dir + "gas_prices.csv")
    if (IS_LOCAL):
        location = pd.read_csv(data_dir + "county_lon_lats.csv")
    else:
        location = prepare_geoopt(data_dir, forecast_weather)

    print("\t> Data Loaded Successfully")

    # initialize data processing pipeline
    FeatureProcessor = FeatureProcessorClass()

    # collate all datasets together
    data = FeatureProcessor(data=train.copy(),
                            client=client.copy(),
                            historical_weather=historical_weather.copy(),
                            forecast_weather=forecast_weather.copy(),
                            electricity=electricity.copy(),
                            gas=gas.copy(),
                            location=location.copy()
                            )

    print("\t> Data is collated successfully")

    # Create all features
    N_day_lags = 15  # Specify how many days we want to go back (at least 2)

    # add time-lagged data to dataset
    df = create_revealed_targets_train(data.copy(),
                                       N_day_lags=N_day_lags)

    print("\t> Lagged data is added successfully")

    # save dataset
    if IS_LOCAL:
        df.to_csv(data_dir + "collated_train.csv", index=False)
        print("\t> Resulting dataframe is saved successfully")


    # reduce memory and model
    df = load_reduce_memory_dataset(data_dir="./data/")

    ''' --------------| CLEANING AND IMPUTATION |-------------- '''

    print(f"\t> Numeric features: #{df.select_dtypes(exclude='object').shape[1]} "
          f"{df.select_dtypes(exclude='object').keys()}")
    print(f"\t> Object features: #{df.select_dtypes(exclude='number').shape[1]} "
          f"{df.select_dtypes(exclude='number').keys()}")

    # Remove empty target row
    target = 'target'
    df = df[df[target].notnull()].reset_index(drop=True)

    # Remove columns for features
    no_features = ['date',
                   'latitude',
                   'longitude',
                   'data_block_id',
                   'row_id',
                   'hours_ahead',
                   'hour_h',
                   ]
    remove_columns = [col for col in df.columns for no_feature in no_features if no_feature in col]
    remove_columns.append(target)
    features = [col for col in df.columns if col not in remove_columns]
    print(f'There are {len(features)} features: {features}')

    ''' --------------| MODELLING |--------------- '''

    model_xgboost(df, features)

    ''' --------------| SUBMISSION |-------------- '''


    # submission
    import enefit

    env = enefit.make_env()
    iter_test = env.iter_test()

    # Reload enefit environment (only in debug mode, otherwise the submission will fail)
    if DEBUG:
        enefit.make_env.__called__ = False
        type(env)._state = type(type(env)._state).__dict__['INIT']
        iter_test = env.iter_test()

    # List of target_revealed dataframes
    previous_revealed_targets = []

    for (test,
         revealed_targets,
         client_test,
         historical_weather_test,
         forecast_weather_test,
         electricity_test,
         gas_test,
         sample_prediction) in iter_test:

        # Rename test set to make consistent with train
        test = test.rename(columns={'prediction_datetime': 'datetime'})

        # Initiate column data_block_id with default value to join on
        id_column = 'data_block_id'

        test[id_column] = 0
        gas_test[id_column] = 0
        electricity_test[id_column] = 0
        historical_weather_test[id_column] = 0
        forecast_weather_test[id_column] = 0
        client_test[id_column] = 0
        revealed_targets[id_column] = 0

        location_test = prepare_geoopt(data_dir, forecast_weather_test)

        data_test = FeatureProcessor(
            data=test,
            client=client_test,
            historical_weather=historical_weather_test,
            forecast_weather=forecast_weather_test,
            electricity=electricity_test,
            gas=gas_test,
            location=location_test
        )

        # Store revealed_targets
        previous_revealed_targets.insert(0, revealed_targets)

        if len(previous_revealed_targets) == N_day_lags:
            previous_revealed_targets.pop()

        # Add previous revealed targets
        df_test = create_revealed_targets_test(data=data_test.copy(),
                                               previous_revealed_targets=previous_revealed_targets.copy(),
                                               N_day_lags=N_day_lags
                                               )

        # Make prediction
        X_test = df_test[features]
        sample_prediction['target'] = clf.predict(X_test)
        env.predict(sample_prediction)