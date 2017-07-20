# Spark-Advanced-Regression-Kaggle
![Alt Text](https://github.com/mkbehbehani/spark-advanced-regression-kaggle/raw/master/demo.gif)
Application which uses machine learning and regression algorithms to predict home prices, used as a term project for a master's course in Big Data and Machine Learning. The goal is to demonstrate the usefulness of various regression algorithms in Spark's machine learning library, [MLlib](https://spark.apache.org/mllib/). To help demonstrate real-world effectiveness  [Kaggle
learning competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) output from various techniques is entered into the competition, resulting in increasing or decreasing accuracy scores. The course for this project focused on Big Data analysis using Hadoop and Spark, so an additional requirement was given to generate predictions using regression techniques that would scale to large datasets on a Spark cluster.

## Training and Test Data
The training and test data given consists of housing data [provided by Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). The data has 79 features, with the addition of Sale Price included on the training data. 

The data is much cleaner than typical real-world data. Only missing value handline was required. After discussion with the instructor, the recommendation was to calculate and [fill missing numeric values with the column mean.]( 
https://github.com/mkbehbehani/spark-advanced-regression-kaggle/blob/master/src/main/scala/HousingAnalyzer.scala#L25)

https://github.com/mkbehbehani/spark-advanced-regression-kaggle/blob/master/source-data/train.csv
