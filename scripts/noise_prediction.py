import findspark
findspark.init()
# Suppressing the possibility of warnings 

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
# Part 1 - Performing ETL activities
# Task 1 - Importing required libraries

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression

import pandas as pd
import os
# Task 2 - Creating a Spark Session

spark = SparkSession.builder.appName("Final Project").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Task 3 - Loading the dataset into the dataframe

df = spark.read.csv("NASA_airfoil_noise_raw.csv", header = True, inferSchema = True)
# Task 4 - Printing the top 5 rows of the dataset

df.show(5)

# Task 5 - Printing the Schema of the dataset
df.printSchema()
# Task 6 - Printing the total number of rows in the dataset

rowcount1 = df.count()
print(rowcount1)
# Task 7 - Dropping all the duplicate rows from the dataset

df = df.dropDuplicates()
df.show()
# Task 8 - Printing the total number of rows in the dataset after dropping the duplicates

rowcount2 = df.count()
print(rowcount2)
# Task 9 - Dropping all null values from the dataset

df = df.dropna()
df.show()
# Task 10 - Printing the total number of dataset

rowcount3 = df.count()
print(rowcount3)
# Task 11 - Rename the column name "SoundLevel" to "SoundLevelDecibels"

df = df.withColumnRenamed("SoundLevel", "SoundLevelDecibels")
df.show()
# Task 12 - Saving the dataframe in parquet format

df.write.mode("overwrite").parquet("NASA_airfoil_noise_cleaned.parquet")
# Part 1 - Evaluation

print("Part 1 - Evaluation.")
print("Total rows =", rowcount1)
print("Total rows after dropping duplicate rows =", rowcount2)
print("Total rows after dropping dupicate rows and null values from the dataset =", rowcount3)

print("Renamed column name =", df.columns[5])
print("NASA_airfoil_noise_cleaned.parquet exists:", os.path.isdir("NASA_airfoil_noise_cleaned.parquet"))
# Part 2 - Creating a Machine Learning Pipeline
# Task 1 - Loading the dataset from NASA_airfoil_noise_cleaned.parquet into the dataframe

df = spark.read.parquet("NASA_airfoil_noise_cleaned.parquet")

# Task 2 = Printing the total number of rows in the dataset
rowcount4 = df.count()

print(rowcount4)
df.show()
# Task 3 - Define the VectorAssembler pipeline stage

assembler = VectorAssembler(inputCols = 
                            ["Frequency", "AngleOfAttack", "ChordLength", "FreeStreamVelocity", 
                             "SuctionSideDisplacement"], 
                            outputCol = "features")
# Task 4 - Define the StandardScaler pipeline stage

scaler = StandardScaler(inputCol = "features", outputCol = "scaledFeatures")
# Task 5 - Define the model creation pipeline stage

lr = LinearRegression(featuresCol="scaledFeatures", labelCol="SoundLevelDecibels")
# Task 6 - Building the pipeline

pipeline = Pipeline(stages = [assembler, scaler, lr])
# Task 7 - Split the data

(trainingData, testingData) = df.randomSplit([0.7, 0.3], seed = 42)
# Task 8 - Fit the pipeline

pipelineModel = pipeline.fit(trainingData)
# Part 2 - Evaluation

print("Print 2 - Evaluation")
print("Total rows = ", rowcount4)
ps = [str(x).split("_")[0] for x in pipeline.getStages()]

print("Pipeline Stage 1 =", ps[0])
print("Pipeline Stage 2 =", ps[1])
print("Pipeline Stage 3 =", ps[2])

print("Lable Column =", lr.getLabelCol())
# Part 3 - Evaluate the Model
# Task 1 - Predict using the model

predictions = pipelineModel.transform(testingData)
# Task 2 - Print the MSE

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SoundLevelDecibels", metricName="mse")
mse = evaluator.evaluate(predictions)
print(mse)
# Task 3 - Print the MAE

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SoundLevelDecibels", metricName="mae")
mae = evaluator.evaluate(predictions)
print(mae)
# Task 4 - Print the R-Squared (r2)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="SoundLevelDecibels", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(r2)
# Part 3 - Evaluation

print("Part 3 - Evaluation")
print("Mean Squared Error =", round(mse,2))
print("Mean Absolute Error =", round(mae,2))
print("R Squared =", round(r2,2))

lrModel = pipelineModel.stages[-1]

print("Intercept =", round(lrModel.intercept,2))
# Part 4 - Persist the Model
# Task 1 - Save the model to the "Final_Project"

pipelineModel.write().overwrite().save("Final_Project")
# Task 2 - Load the Model from "Final_Project"

loadedPipelineModel = PipelineModel.load("Final_Project")
# Task 3 - Make predictions using the loaded model on the testdata

predictions = loadedPipelineModel.transform(testingData)
# Task 4 - Show the predictions

predictions.select("SoundLevelDecibels", "prediction").show(5)
# Part 4 - Evaluation

print("Part 4 - Evaluation")

loadedmodel = loadedPipelineModel.stages[-1]
totalstages = len(loadedPipelineModel.stages)
inputcolumns = loadedPipelineModel.stages[0].getInputCols()

print("Number of stages in the pipeline = ", totalstages)
for i,j in zip(inputcolumns, loadedmodel.coefficients):
    print(f"Coefficient for {i} is {round(j,4)}")
# Stop Spark Session

spark.stop()

