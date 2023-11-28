import pandas as pd
import numpy as np
import json
from pathlib import Path

from geopy.geocoders import Nominatim

def prepare_geoopt(data_dir):
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

    forecast_weather = pd.read_csv("./data/"+"forecast_weather.csv")

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


    print(f"Saving dataframe with {new_df.shape} shape")

    new_df.to_csv(data_dir+"county_lon_lats.csv",index=False)

def preprocess_datasets(data_dir):
    train = pd.read_csv(data_dir + "train.csv")
    client = pd.read_csv(data_dir + "client.csv")
    historical_weather = pd.read_csv(data_dir + "historical_weather.csv")
    forecast_weather = pd.read_csv(data_dir + "forecast_weather.csv")
    electricity = pd.read_csv(data_dir + "electricity_prices.csv")
    gas = pd.read_csv(data_dir + "gas_prices.csv")
    location = pd.read_csv(data_dir + "county_lon_lats.csv")
    print("Data Loaded Successfully")
def load_process():
    # data_dir = "./data/"
    # train = pd.read_csv(data_dir + "train.csv")
    # client = pd.read_csv(data_dir + "client.csv")
    # historical_weather = pd.read_csv(data_dir + "historical_weather.csv")
    # forecast_weather = pd.read_csv(data_dir + "forecast_weather.csv")
    # electricity = pd.read_csv(data_dir + "electricity_prices.csv")
    # gas = pd.read_csv(data_dir + "gas_prices.csv")
    # print("Data Loaded Successfully")
    prepare_geoopt(data_dir = "./data/")
    # preprocess_datasets(data_dir = "./data/")


if __name__ == '__main__':
    load_process()