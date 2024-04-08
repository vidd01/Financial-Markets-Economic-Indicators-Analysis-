import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
import datetime
import logging
from azure.cosmos import CosmosClient, exceptions

# logging.basicConfig(level=logging.INFO)

#connection variables
cosmos_url = 'https://capstonesosmosstorage.documents.azure.com:443/'
cosmos_key = '6nWf2nyesaeRndaId1iR4QGkFqzLxj72moSOGBxSM7GqVRWgT6qteISZMUeGYgb7eYpm6aLs6XhzACDbSXeLFg=='

#macro economic series - FRED - variables
#Canada
canada_gdp = 'NGDPRXDCCAA'
canada_bond_rates = 'IRLTLT01CAM156N'
canada_unemployment = 'LRUNTTTTCAM156S'
canada_inflation = 'CPALCY01CAM661N'
canada_bank_rates = 'IR3TIB01CAM156N'
#India
india_gdp = 'MKTGDPINA646NWDB'
india_bond_rates = 'INDIRLTLT01STM'
india_inflation = 'INDCPIALLMINMEI'
india_bank_rates = 'INDIR3TIB01STM'
#United States
usa_gdp = 'GDPA'
usa_inflation = 'CPIAUCSL'
usa_unemployment = 'UNRATE'
usa_bank_rates = 'FEDFUNDS'
usa_bond_rates = 'IRLTLT01USM156N'
#Indices
sp500 = '^GSPC'
dowjones = '^DJI'
nasdaq = '^IXIC'
nifty50 = '^NSEI'
tsx = '^GSPTSE'


def get_macro_data_and_upload_to_cosmos(series, id , partition_key, db_name, db_cn_name, cosmos_url, cosmos_key):
    # Start and end date to download max available data
    start_date = datetime.datetime(1900, 1, 1)
    end_date = datetime.datetime.now()

    # Get the data from FRED
    if series == 'MKTGDPINA646NWDB':
        data = pdr.get_data_fred(series, start_date, end_date) / 1e12
    elif series == 'NGDPRXDCCAA':
        data = pdr.get_data_fred(series, start_date, end_date) / 1e6
    elif series == 'GDPA':
        data = pdr.get_data_fred(series, start_date, end_date) / 1000
    else:
        data = pdr.get_data_fred(series, start_date, end_date)
        

    # Convert date to string and create a dictionary
    data.reset_index(inplace=True)
    data['DATE'] = pd.to_datetime(data['DATE']).dt.strftime('%Y-%m-%d')
    data_dict = dict(zip(data['DATE'], data[series]))

    # Add necessary fields to the dictionary
    data_dict['id'] = id  # Using series as the ID
    data_dict['partitionKey'] = partition_key

    current_time_utc = datetime.datetime.utcnow()
    utc_offset = datetime.timedelta(hours=-4)
    current_time_utc_minus_4 = current_time_utc + utc_offset
    data_dict['updateDateTime'] = current_time_utc_minus_4.strftime('%Y-%m-%d %H:%M:%S')

    # Connect to Cosmos DB
    client = CosmosClient(cosmos_url, credential=cosmos_key)
    database = client.get_database_client(db_name)
    container = database.get_container_client(db_cn_name)

    try:
        existing_item = container.read_item(item=data_dict['id'], partition_key=data_dict['partitionKey'])
        existing_item.update(data_dict)
        container.replace_item(item=data_dict['id'], body=existing_item)
        # logging.info(f"Item '{partition_key} - {series}' updated successfully.")
        print(f"Item '{db_name} - {partition_key}' updated successfully.")
    except exceptions.CosmosHttpResponseError as e:
        # logging.error(f"Failed to update item '{series}': {e}")
        print(f"Failed to update item '{series}': {e}")


def get_financial_data_and_upload_to_cosmos(series, id , partition_key, db_name, db_cn_name, cosmos_url, cosmos_key):
    # Start and end date to download max available data
    start_date = datetime.datetime(1900, 1, 1)
    end_date = datetime.datetime.now()
    
    data = yf.download(series, start=start_date, end=end_date, interval='1mo')
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
    data = data.set_index('Date')
    data_dict = data.to_dict(orient='index')
    
    # Add necessary fields to the dictionary
    data_dict['id'] = id  # Using series as the ID
    data_dict['partitionKey'] = partition_key

    current_time_utc = datetime.datetime.utcnow()
    utc_offset = datetime.timedelta(hours=-4)
    current_time_utc_minus_4 = current_time_utc + utc_offset
    data_dict['updateDateTime'] = current_time_utc_minus_4.strftime('%Y-%m-%d %H:%M:%S')

    # Connect to Cosmos DB
    client = CosmosClient(cosmos_url, credential=cosmos_key)
    database = client.get_database_client(db_name)
    container = database.get_container_client(db_cn_name)

    try:
        existing_item = container.read_item(item=data_dict['id'], partition_key=data_dict['partitionKey'])
        existing_item.update(data_dict)
        container.replace_item(item=data_dict['id'], body=existing_item)
        logging.info(f"Item '{partition_key} - {series}' updated successfully.")
        print(f"Item '{partition_key} - {series}' updated successfully.")
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f"Failed to update item '{series}': {e}")
        print(f"Failed to update item '{series}': {e}")




if __name__ == "__main__":
    # Canada data push
    # get_macro_data_and_upload_to_cosmos(series=canada_gdp,
    #                                     id='1',partition_key='gdp',
    #                                     db_name='CanadaDB',db_cn_name='Canada',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=canada_inflation,
    #                                     id='2',partition_key='inflation',
    #                                     db_name='CanadaDB',db_cn_name='Canada',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=canada_bond_rates,
    #                                     id='3',partition_key='bondRates',
    #                                     db_name='CanadaDB',db_cn_name='Canada',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=canada_unemployment,
    #                                     id='4',partition_key='unemployment',
    #                                     db_name='CanadaDB',db_cn_name='Canada',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=canada_bank_rates,
    #                                     id='5',partition_key='bankRates',
    #                                     db_name='CanadaDB',db_cn_name='Canada',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)

    # # USA data push
    # get_macro_data_and_upload_to_cosmos(series=usa_gdp,
    #                                     id='1',partition_key='gdp',
    #                                     db_name='UnitedStatesDB',db_cn_name='USA',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=usa_inflation,
    #                                     id='2',partition_key='inflation',
    #                                     db_name='UnitedStatesDB',db_cn_name='USA',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=usa_bond_rates,
    #                                     id='3',partition_key='bondRates',
    #                                     db_name='UnitedStatesDB',db_cn_name='USA',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=usa_unemployment,
    #                                     id='4',partition_key='unemployment',
    #                                     db_name='UnitedStatesDB',db_cn_name='USA',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=usa_bank_rates,
    #                                     id='5',partition_key='bankRates',
    #                                     db_name='UnitedStatesDB',db_cn_name='USA',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)

    # # India data push
    # get_macro_data_and_upload_to_cosmos(series=india_gdp,
    #                                     id='1',partition_key='gdp',
    #                                     db_name='IndiaDB',db_cn_name='India',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=india_inflation,
    #                                     id='2',partition_key='inflation',
    #                                     db_name='IndiaDB',db_cn_name='India',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=india_bond_rates,
    #                                     id='3',partition_key='bondRates',
    #                                     db_name='IndiaDB',db_cn_name='India',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    # get_macro_data_and_upload_to_cosmos(series=india_bank_rates,
    #                                     id='5',partition_key='bankRates',
    #                                     db_name='IndiaDB',db_cn_name='India',
    #                                     cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    
    #financial data
    get_financial_data_and_upload_to_cosmos(series=sp500,
                                            id='1',partition_key='sp500',
                                            db_name='Indices',db_cn_name='Indices',
                                            cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    get_financial_data_and_upload_to_cosmos(series=dowjones,
                                            id='2',partition_key='dowjones',
                                            db_name='Indices',db_cn_name='Indices',
                                            cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    get_financial_data_and_upload_to_cosmos(series=nasdaq,
                                            id='3',partition_key='nasdaq',
                                            db_name='Indices',db_cn_name='Indices',
                                            cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    get_financial_data_and_upload_to_cosmos(series=nifty50,
                                            id='4',partition_key='nifty50',
                                            db_name='Indices',db_cn_name='Indices',
                                            cosmos_url=cosmos_url,cosmos_key=cosmos_key)
    get_financial_data_and_upload_to_cosmos(series=tsx,
                                            id='5',partition_key='tsx',
                                            db_name='Indices',db_cn_name='Indices',
                                            cosmos_url=cosmos_url,cosmos_key=cosmos_key)



