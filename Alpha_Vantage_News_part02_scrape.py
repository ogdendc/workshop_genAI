# Databricks notebook source
# MAGIC %md ## This program picks up where '...part01' leaves off:
# MAGIC > ### taking the list of urls, webscraping to pull articles,
# MAGIC > ### outputting html type text that fits nicely into the RAG demo,
# MAGIC > ### and outputting cleaner text for AI Functions demo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### reading in the table of urls created in the part01 program:

# COMMAND ----------

# MAGIC %md
# MAGIC #### (if not using a ML runtime, will need to install the Huggingface 'datasets' library)

# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## NOTE:
# MAGIC > ### You can read this list from Huggingface, or...
# MAGIC > ### you can use your own version created from running the "...part01" program.

# COMMAND ----------

# reading the dataset from Huggingface (requires a ML DBR 13+ or pip install 'datasets'):

from datasets import load_dataset

dataset = load_dataset("ogdendc/alpha_vantage_list")
df_unioned = spark.createDataFrame(dataset["train"].to_pandas())
display(df_unioned)

# COMMAND ----------

# MAGIC %md
# MAGIC # NOTICE:
# MAGIC > ## Comment the cell below to run this on the full input list of articles.
# MAGIC > ## This is here for testing purposes, since doing the webcrawling on the full list could take hours.

# COMMAND ----------

# Note to user:  use this optional filter to limit records being processed, for testing purposes:

#df_unioned = df_unioned.filter("ticker in ('MSFT', 'AAPL')")

# COMMAND ----------

# MAGIC %md #The data from Alpha Vantage has highly summarized text.  Soooo...
# MAGIC > ## crawling each identified site to retrieve full article:

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql.types import StringType
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
#Add retries with backoff to avoid 429 while fetching the doc
retries = Retry(
    total=3,
    backoff_factor=3,
    status_forcelist=[429],
)

def download_documentation_articles(max_documents=None):

    # Create DataFrame from URLs
    df_urls = df_unioned #.select("url").repartition(10)

    # Pandas UDF to fetch HTML content for a batch of URLs
    @pandas_udf("string")
    def fetch_html_udf(urls: pd.Series) -> pd.Series:
        adapter = HTTPAdapter(max_retries=retries)
        http = requests.Session()
        http.mount("http://", adapter)
        http.mount("https://", adapter)
        def fetch_html(url):
            try:
                #response = http.get(url)
                # added user-agent to supposedly tell website who is accessing, which reduced null responses for raw content
                response = http.get(url, headers={'user-agent': 'my-app/1.1.1'}) 
                if response.status_code == 200:
                    return response.content
            except requests.RequestException:
                return None
            return None

        with ThreadPoolExecutor(max_workers=200) as executor:
            results = list(executor.map(fetch_html, urls))
        return pd.Series(results)

    # Pandas UDF to process HTML content and extract text
    @pandas_udf("string")
    def download_web_page_udf(html_contents: pd.Series) -> pd.Series:
        def extract_text(html_content):
            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                article_div = soup.find("div", itemprop="articleBody")
                # adding more conditions to try to retrieve less nulls, based on investigations below
                comtext_div = soup.find("div", id="comtext")
                if article_div:
                    return str(article_div).strip()
                elif comtext_div:
                    return str(comtext_div).strip()
            return None

        return html_contents.apply(extract_text)

    # Apply UDFs to DataFrame
    df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
    final_df = df_with_html.withColumn("text", download_web_page_udf("html_content"))

    # Select and filter non-null results
    #final_df = final_df.select("url", "text").filter("text IS NOT NULL").cache()
    if final_df.isEmpty():
      raise Exception("Dataframe is empty, couldn't download documentation, please check sitemap status.")

    return final_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## running the above code to scrape all sites (this could take hours depending on size of the list)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

df_unioned_w_text = download_documentation_articles()
display(df_unioned_w_text)

# COMMAND ----------

# MAGIC %md ## removing any records with null value for 'text'
# MAGIC > ### (which would be the result of an unsuccessful webscrape)

# COMMAND ----------

# removing records with unsuccessful webscrape results

df_unioned_w_text = df_unioned_w_text.filter("text IS NOT NULL")

# COMMAND ----------

# MAGIC %md
# MAGIC ## adding a cleaner version of the text, stripping the html bits

# COMMAND ----------

from pyspark.sql.functions import col, udf

# Define a UDF to apply the BeautifulSoup get_text function
# NOTE: had to specify 'html.parser' to avoid an error (i.e. maybe default 'lxml' parser not a great idea?)
soup_udf = udf(lambda text: BeautifulSoup(text, "html.parser").get_text(), StringType())

# Apply the UDF to create a new column "text_clean"
df_unioned_w_text = df_unioned_w_text.withColumn("text_clean", soup_udf(col("text")))

display(df_unioned_w_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Truncating the text output to remove the extreme outliers with very long text strings / word counts:

# COMMAND ----------

from pyspark.sql.functions import substring, col

df_unioned_w_text = df_unioned_w_text.withColumn("text", substring(col("text"), 1, 50000)).\
  withColumn("text_clean", substring(col("text_clean"), 1, 50000))

display(df_unioned_w_text)

# COMMAND ----------

# MAGIC %md #Notes:
# MAGIC > ## The 'url' and 'text' columns provide plug-n-play option for converting the standard RAG demo into a FinServ focused RAG demo.
# MAGIC > ## The 'text_clean' column is great option to trialing some Databricks AI functions.
