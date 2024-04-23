# Databricks notebook source
# MAGIC %md ## This program picks up where '...part02' leaves off:
# MAGIC > ### reading in the cleaned text and analyzing with Databricks AI Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## use DBR 15+ to support AI functions
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/large-language-models/ai-functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## NOTE:
# MAGIC > ### You can read this list from Huggingface, or...
# MAGIC > ### you can use your own version created from running the "...part02" program.

# COMMAND ----------

# reading the dataset from Huggingface (requires a ML DBR 13+ or pip install 'datasets'):

from datasets import load_dataset

dataset = load_dataset("ogdendc/alpha_vantage_scraped")
df_unioned = spark.createDataFrame(dataset["train"].to_pandas())
display(df_unioned)

# COMMAND ----------

# Create a temporary view on the dataframe to enable SQL
df_unioned.createOrReplaceTempView("temp_alphavantage_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ### the list of tickers to pick from:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT ticker
# MAGIC FROM temp_alphavantage_view
# MAGIC   ;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   text_clean,
# MAGIC   ai_analyze_sentiment(text_clean)                                            AS sentiment,
# MAGIC   ai_classify(text_clean, array('investments', 'litigation', 'AI', 'other'))  AS news_category,
# MAGIC   ai_classify(text_clean, array('bearish', 'bullish', 'neutral', 'hold'))     AS investment_category,
# MAGIC   ai_extract(text_clean, array('location', 'organization'))                   AS entities,
# MAGIC   ai_gen(
# MAGIC         'You are a concise financial advisor. Provide investment advice, in less than 50 words, based on this news article: ' || text_clean
# MAGIC         )                                                                     AS investment_advice,
# MAGIC   ai_mask(text_clean, array('person', 'company'))                             AS text_masked,
# MAGIC   ai_summarize(text_clean, 20)                                                AS text_sum_20words,
# MAGIC   ai_query('databricks-dbrx-instruct', CONCAT('Concisely summarize this article:\n', text_clean)) AS dbrx_text_sum,
# MAGIC   ai_query('databricks-mixtral-8x7b-instruct', CONCAT('Identify and list the entities in this text:\n', text_clean))     AS mxtrl_entities
# MAGIC FROM
# MAGIC   temp_alphavantage_view
# MAGIC WHERE ticker = "MSFT"
# MAGIC LIMIT 20
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC ## More AI Functions demonstrated below:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     ai_fix_grammar("This sentence be very poor written and definite need have fixed for gooder clarity.") AS gooder_text
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_similarity("This incomplete sentence about Databricks on AWS", "is this similar to this other incomplete sentence about Azure Databricks.");
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_translate('Me encanta hacer cosas geniales de IA.', 'en');

# COMMAND ----------


