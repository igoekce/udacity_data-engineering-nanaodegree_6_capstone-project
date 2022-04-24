#!/usr/bin/env python

# Do all imports and installs here and initiate spark
import pandas as pd
import datetime as dt
import numpy as np

from pyspark.sql import SparkSession
spark = SparkSession.builder.config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0").getOrCreate()

from pyspark.sql.functions import first
from pyspark.sql.functions import upper, col
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType
from pyspark.sql.functions import udf, date_format


############## us-cities-demographics ##############
us_spark=spark.read.csv("./data/us-cities-demographics.csv", sep=';', header=True)

# Creating 'us_race_count' dataset
us_race_count=(us_spark.select("city","state code","Race","count")
    .groupby(us_spark.City, "state code")
    .pivot("Race")
    .agg(first("Count")))

uscols=["Number of Veterans","Race","Count"]
us=us_spark.drop(*uscols).dropDuplicates()

us=us.join(us_race_count, ["city","state code"])

# Change `state code` column name to `state_code` and other similar problems to avoid parquet complications
us=us.select('City', col('State Code').alias('State_Code'), 'State', col('Median Age').alias('Median_age'),
     col('Male Population').alias('Male_Pop'), col('Female Population').alias('Fem_Pop'), 
        col('Total Population').alias('Ttl_Pop'), 'Foreign-born', 
          col('Average Household Size').alias('Avg_Household_Size'),
             col('American Indian and Alaska Native').alias('Native_Pop'), 
                 col('Asian').alias('Asian_Pop'), 
                    col('Black or African-American').alias('Black_Pop'), 
                      col('Hispanic or Latino').alias('Latino_Pop'), 
                        col('White').alias('White_Pop'))

us=us.drop("state")

### WRITE ###
us.write.mode('overwrite').parquet("./data/us_cities_demographics.parquet")


############## sas_data ##############
i94_spark=spark.read.parquet("./data/sas_data")

i94_spark=i94_spark.select(col("i94res").cast(IntegerType()),col("i94port"),
                           col("arrdate").cast(IntegerType()), \
                           col("i94mode").cast(IntegerType()),col("depdate").cast(IntegerType()),
                           col("i94bir").cast(IntegerType()),col("i94visa").cast(IntegerType()), 
                           col("count").cast(IntegerType()), \
                              "gender",col("admnum").cast(LongType()))

i94_spark=i94_spark.dropDuplicates()


############## sas labels ##############
with open('./data/I94_SAS_Labels_Descriptions.SAS') as f:
    f_content = f.read()
    f_content = f_content.replace('\t', '')

def code_mapper(file, idx):
    f_content2 = f_content[f_content.index(idx):]
    f_content2 = f_content2[:f_content2.index(';')].split('\n')
    f_content2 = [i.replace("'", "") for i in f_content2]
    dic = [i.split('=') for i in f_content2[1:]]
    dic = dict([i[0].strip(), i[1].strip()] for i in dic if len(i) == 2)
    return dic

i94cit_res = code_mapper(f_content, "i94cntyl")
i94port = code_mapper(f_content, "i94prtl")
i94mode = code_mapper(f_content, "i94model")
i94addr = code_mapper(f_content, "i94addrl")
i94visa = {'1':'Business',  '2': 'Pleasure', '3' : 'Student'}


# Start processing the I94_SAS_Labels_Description

############## i94mode ##############
# Create i94mode list
i94mode_data =[[1,'Air'],[2,'Sea'],[3,'Land'],[9,'Not reported']]
i94mode=spark.createDataFrame(i94mode_data)

### WRITE ###
i94mode.write.mode("overwrite").parquet('./data/i94mode.parquet')

############## i94port ##############
df = pd.DataFrame(list(i94port.items()),columns = ['id','port_city']) 
i94port_df = pd.concat([df, df['port_city'].str.split(', ', expand=True)], axis=1).drop(columns=['port_city', 2]).rename(columns={0: 'city', 1: 'state'}).set_index('id')
df.to_csv('./data/i94port.csv')

i94port_df = pd.read_csv('./data/i94port.csv')
i94port_data=i94port_df.values.tolist()

i94port_schema = StructType([
    StructField('id', StringType(), True),
    StructField('port_city', StringType(), True),
    StructField('port_state', StringType(), True)
])
i94port=spark.createDataFrame(i94port_data, i94port_schema)

### WRITE ###
i94port.write.mode('overwrite').parquet('./data/i94port.parquet')

############## i94res ##############
i94res_df = pd.DataFrame(list(i94cit_res.items()),columns = ['id','country']) 
i94res_data=i94res_df.values.tolist()
i94res_schema = StructType([
    StructField('id', StringType(), True),
    StructField('country', StringType(), True)
])
i94res=spark.createDataFrame(i94res_data, i94res_schema)

### WRITE ###
i94res.write.mode('overwrite').parquet('./data/i94res.parquet')


############## i94visa_data ##############
i94visa_data = [[1, 'Business'], [2, 'Pleasure'], [3, 'Student']]
i94visa=spark.createDataFrame(i94visa_data)


### WRITE ###
i94visa.write.mode('overwrite').parquet('./data/i94visa.parquet')


# enrich i94 df
i94_spark=i94_spark.join(i94port, i94_spark.i94port==i94port.id, how='left')
i94_spark=i94_spark.drop("id")

i94non_immigrant_port_entry=i94_spark.join(us, (upper(i94_spark.port_city)==upper(us.City)) & \
                                           (upper(i94_spark.port_state)==upper(us.State_Code)), how='left')

i94non_immigrant_port_entry=i94non_immigrant_port_entry.drop("City","State_Code")


############## date ##############
get_date = udf(lambda x: (dt.datetime(1960, 1, 1).date() + dt.timedelta(x)).isoformat() if x else None)
i94non_immigrant_port_entry = i94non_immigrant_port_entry.withColumn("arrival_date", get_date(i94non_immigrant_port_entry.arrdate))

i94date=i94non_immigrant_port_entry.select(col('arrdate').alias('arrival_sasdate'),
                                   col('arrival_date').alias('arrival_iso_date'),
                                   date_format('arrival_date','M').alias('arrival_month'),
                                   date_format('arrival_date','E').alias('arrival_dayofweek'), 
                                   date_format('arrival_date', 'y').alias('arrival_year'), 
                                   date_format('arrival_date', 'd').alias('arrival_day'),
                                  date_format('arrival_date','w').alias('arrival_weekofyear')).dropDuplicates()

### WRITE ###
i94non_immigrant_port_entry.drop('arrival_date').write.mode("overwrite").parquet('./data/i94non_immigrant_port_entry.parquet')


# i94date
i94date.createOrReplaceTempView("i94date_table")

i94date_season=spark.sql('''select arrival_sasdate,
                         arrival_iso_date,
                         arrival_month,
                         arrival_dayofweek,
                         arrival_year,
                         arrival_day,
                         arrival_weekofyear,
                         CASE WHEN arrival_month IN (12, 1, 2) THEN 'winter' 
                                WHEN arrival_month IN (3, 4, 5) THEN 'spring' 
                                WHEN arrival_month IN (6, 7, 8) THEN 'summer' 
                                ELSE 'autumn' 
                         END AS date_season from i94date_table''')

#### Write ####
i94date_season.write.mode("overwrite").partitionBy("arrival_year", "arrival_month").parquet('./data/i94date.parquet')
