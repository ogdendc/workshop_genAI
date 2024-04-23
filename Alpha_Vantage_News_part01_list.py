# Databricks notebook source
# MAGIC %md # Creating a list of urls for market news articles
# MAGIC > ## using the Alphas Vantage market news API:  
# MAGIC >>> ### https://www.alphavantage.co/documentation/#intelligence
# MAGIC > ## limited to 25 per day to avoid hitting API limits

# COMMAND ----------

# MAGIC %md
# MAGIC # NOTE:
# MAGIC > ## User inputs are needed for this code to run.
# MAGIC > ## 1. A "json_file_save_location" that points to a path to which to dump json files from the API pull.
# MAGIC >>> ### (on file will be saved out for each ticker)
# MAGIC > ## 2. "API_key".  This can be obtained for free at https://www.alphavantage.co/support/#api-key

# COMMAND ----------

# change to your own location
json_file_save_location = "/Volumes/main/XYZ/ABC"

# change to your own key
API_key = "1234567ABCDEFG"

# COMMAND ----------

# MAGIC %md # WARNING!!!
# MAGIC > # run this next line only to REMOVE every existing file in the Volume
# MAGIC > ## as a bit of cleanup, if starting this whole process from scratch

# COMMAND ----------

#dbutils.fs.rm(f"{json_file_save_location}", recurse=True)

# COMMAND ----------

# MAGIC %md ## read in a list of tickers for which to pull the market news
# MAGIC > ### (from a csv file stored in Github)
# MAGIC

# COMMAND ----------

# https://github.com/ogdendc/workshop_genAI/blob/main/stock_ticker_list_500_from_NASDAQ.csv

import pandas as pd
pd_df = pd.read_csv("https://raw.githubusercontent.com/ogdendc/workshop_genai_data/main/stock_ticker_list_500_from_NASDAQ.csv", header=0)
display(pd_df)

# COMMAND ----------

# MAGIC %md ## creating list of tickers for which to retrieve the json file via the API
# MAGIC > ### code below is set to pull the first 25 from the list of tickers above

# COMMAND ----------

#reading a range of rows
ticker_list = pd_df['ticker'].iloc[[*range(0, 25)]].tolist()
print(ticker_list)

# COMMAND ----------

# MAGIC %md ## function to make the API call to Alpha Vantage Global News API

# COMMAND ----------

import requests
import json

def make_api_call_and_save(url, filename):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'w') as file:
                json_data = response.json()
                file.write(json.dumps(json_data))
            print(f"Data saved to {filename}")
        else:
            print(f"Error: Unable to fetch data, status code {response.status_code}")
    except requests.RequestException as e:
        print(f"Error: {e}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## for each ticker in the list, pull the json from Alpha Vantage market news API
# MAGIC > ## and save the json files to specified location (above)

# COMMAND ----------

for ticker in ticker_list:
    api_url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={}&apikey={}&limit=1000".format(ticker, API_key)  
    save_json_filename = "{}/news_data_{}.json".format(json_file_save_location, ticker)
    # Make the API call and save the results
    make_api_call_and_save(api_url, save_json_filename)

# COMMAND ----------

# MAGIC %md ## NOTE:
# MAGIC > ### some of the tickers have inconsistent json file formats
# MAGIC > ### and those tickers are dropped due to file format inconsistency

# COMMAND ----------

removal_list = ['GOOGL', 'CMCSA']

ticker_list = [k for k in ticker_list if k not in removal_list]

# COMMAND ----------

# MAGIC %md ## read each json, union into single dataframe, then explode the columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### read each dataframe into a list of dataframes

# COMMAND ----------

from pyspark.sql.functions import lit

# creating list of dataframes created
df_list = []

for ticker in ticker_list:
  filename = "{}/news_data_{}.json".format(json_file_save_location, ticker)
  locals()['df_' + ticker] = spark.read.json(filename).withColumn("ticker", lit(ticker))
  df_list.append(locals()['df_' + ticker])

# COMMAND ----------

# MAGIC %md
# MAGIC ### union all dataframes into one

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame

df_unioned = reduce(DataFrame.unionAll, df_list) 

display(df_unioned)

# COMMAND ----------

# MAGIC %md ### explode and select desired columns

# COMMAND ----------

from pyspark.sql.functions import explode, col

# using the 'explode' function to parse out the elements of the data array

df_unioned = df_unioned.select("ticker", explode("feed")).select("ticker", "col.*").select("ticker", "source", "source_domain", "url", "title", col("authors")[0].alias("author"), "time_published", "summary")

display(df_unioned)

# COMMAND ----------

# MAGIC %md
# MAGIC # NOTE:
# MAGIC > ## a modified version of this program was used to read-in two "batches" of API calls
# MAGIC > ## and the resulting list was saved to Huggingface at ogdendc/alpha_vantage_list

# COMMAND ----------

# reading the dataset from Huggingface (requires a ML DBR 13+) or a pip install of 'datasets':

# from datasets import load_dataset

# dataset = load_dataset("ogdendc/alpha_vantage_list")
# spark_df = spark.createDataFrame(dataset["train"].to_pandas())
# display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Next step:
# MAGIC > ## part02 program to read in the full list, scrape all the websites...
# MAGIC > ## to extract the full articles from the list of urls above
