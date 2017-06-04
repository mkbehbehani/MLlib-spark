
import java.util.Calendar

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.max
import org.apache.log4j.{Level, Logger}

object HousingAnalyzer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("PipelineExample")
      .getOrCreate()
    import spark.implicits._
    val trainingSchema = StructType(Array(
      StructField("Id", IntegerType, true),
      StructField("SalePrice", IntegerType, true),
      StructField("MSSubClass", IntegerType, true),
      StructField("LotArea", IntegerType, true),
      StructField("OverallQual", IntegerType, true),
      StructField("OverallCond", IntegerType, true),
      StructField("YearBuilt", IntegerType, true),
      StructField("YearRemodAdd", IntegerType, true),
      StructField("BsmtFinSF1", IntegerType, true),
      StructField("BsmtFinSF2", IntegerType, true),
      StructField("BsmtUnfSF", IntegerType, true),
      StructField("TotalBsmtSF", IntegerType, true),
      StructField("1stFlrSF", IntegerType, true),
      StructField("2ndFlrSF", IntegerType, true),
      StructField("LowQualFinSF", IntegerType, true),
      StructField("GrLivArea", IntegerType, true),
      StructField("BsmtFullBath", IntegerType, true),
      StructField("BsmtHalfBath", IntegerType, true),
      StructField("FullBath", IntegerType, true),
      StructField("HalfBath", IntegerType, true),
      StructField("BedroomAbvGr", IntegerType, true),
      StructField("KitchenAbvGr", IntegerType, true),
      StructField("TotRmsAbvGrd", IntegerType, true),
      StructField("Fireplaces", IntegerType, true),
      StructField("GarageCars", IntegerType, true),
      StructField("GarageArea", IntegerType, true),
      StructField("WoodDeckSF", IntegerType, true),
      StructField("OpenPorchSF", IntegerType, true),
      StructField("EnclosedPorch", IntegerType, true),
      StructField("3SsnPorch", IntegerType, true),
      StructField("ScreenPorch", IntegerType, true),
      StructField("PoolArea", IntegerType, true),
      StructField("MoSold", IntegerType, true),
      StructField("YrSold", IntegerType, true)
    ))

    val testSchema = StructType(Array(
      StructField("Id", IntegerType, true),
      StructField("MSSubClass", IntegerType, true),
      StructField("LotArea", IntegerType, true),
      StructField("OverallQual", IntegerType, true),
      StructField("OverallCond", IntegerType, true),
      StructField("YearBuilt", IntegerType, true),
      StructField("YearRemodAdd", IntegerType, true),
      StructField("BsmtFinSF1", IntegerType, true),
      StructField("BsmtFinSF2", IntegerType, true),
      StructField("BsmtUnfSF", IntegerType, true),
      StructField("TotalBsmtSF", IntegerType, true),
      StructField("1stFlrSF", IntegerType, true),
      StructField("2ndFlrSF", IntegerType, true),
      StructField("LowQualFinSF", IntegerType, true),
      StructField("GrLivArea", IntegerType, true),
      StructField("BsmtFullBath", IntegerType, true),
      StructField("BsmtHalfBath", IntegerType, true),
      StructField("FullBath", IntegerType, true),
      StructField("HalfBath", IntegerType, true),
      StructField("BedroomAbvGr", IntegerType, true),
      StructField("KitchenAbvGr", IntegerType, true),
      StructField("TotRmsAbvGrd", IntegerType, true),
      StructField("Fireplaces", IntegerType, true),
      StructField("GarageCars", IntegerType, true),
      StructField("GarageArea", IntegerType, true),
      StructField("WoodDeckSF", IntegerType, true),
      StructField("OpenPorchSF", IntegerType, true),
      StructField("EnclosedPorch", IntegerType, true),
      StructField("3SsnPorch", IntegerType, true),
      StructField("ScreenPorch", IntegerType, true),
      StructField("PoolArea", IntegerType, true),
      StructField("MoSold", IntegerType, true),
      StructField("YrSold", IntegerType, true)
    ))
    val rawTrainingData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/train.csv")
    val rawTestData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/test.csv")

    val countFeaturesTrainingDF = rawTrainingData.select("Id","SalePrice","MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    val nullCorrectedTrainingDf = countFeaturesTrainingDF.na.fill(countFeaturesTrainingDF.columns.zip(
      countFeaturesTrainingDF.select(countFeaturesTrainingDF.columns.map(mean): _*).first.toSeq
    ).toMap)

    val countFeaturesTestDF = rawTestData.select("Id","MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    countFeaturesTrainingDF.show(2)
    val nullCorrectedTestDf = countFeaturesTestDF.na.fill(countFeaturesTestDF.columns.zip(
      countFeaturesTestDF.select(countFeaturesTestDF.columns.map(mean): _*).first.toSeq
    ).toMap)


    val featureColumns = Array("MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    val labelColumn = "SalePrice"

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val castTrainingData = spark.createDataFrame(nullCorrectedTrainingDf.rdd, trainingSchema)
    val castTestData = spark.createDataFrame(nullCorrectedTestDf.rdd, testSchema)

    val featurizedTrainingData = assembler.transform(castTrainingData).select("Id","SalePrice", "features")
    val featurizedTestData = assembler.transform(castTestData).select("Id","features")


    val trainingData = featurizedTrainingData
    val testData = featurizedTestData
//    val Array(trainingData, testData) = featurizedTrainingData.randomSplit(Array(0.8, 0.2))

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("SalePrice")
      .setFeaturesCol("features")

    // Chain indexer and tree in a Pipeline.
    val pipeline2 = new Pipeline()
      .setStages(Array(dt))


    // Train model. This also runs the indexer.
    val model2 = pipeline2.fit(trainingData)

    // Make predictions.
    val predictions2 = model2.transform(testData)
    predictions2.show(3)
    // Select example rows to display.
//    predictions2.withColumnRenamed("prediction", "SalePrice").select("Id","SalePrice").coalesce(1).write.option("header", "true").csv(System.getProperty("user.dir") + "/housing-predictions/" + Calendar.getInstance().getTime.toString)
    // Select (prediction, true label) and compute test error.
//    val evaluator2 = new RegressionEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("rmse")
//    val rmse2 = evaluator2.evaluate(predictions2)
//
//    val treeModel = model2.stages(0).asInstanceOf[DecisionTreeRegressionModel]
//    println("Learned regression tree model:\n" + treeModel.toDebugString)

    // Train a RandomForest model.
//    val rf = new RandomForestRegressor()
//      .setLabelCol("SalePrice")
//      .setFeaturesCol("features")
//
//    // Chain indexer and forest in a Pipeline.
//    val rfPipeline = new Pipeline()
//      .setStages(Array(rf))
//
//    // Train model. This also runs the indexer.
//    val rfFittedModel = rfPipeline.fit(trainingData)
//
//    // Make predictions.
//    val rfPredictions = rfFittedModel.transform(testData)
//
//    // Select example rows to display.
//    rfPredictions.select("prediction", "SalePrice", "features").show(5)
//
//    // Select (prediction, true label) and compute test error.
//    val rfevaluator = new RegressionEvaluator()
//      .setLabelCol("SalePrice")
//      .setPredictionCol("prediction")
//      .setMetricName("rmse")
//    val RFrmse = rfevaluator.evaluate(rfPredictions)
//
//    val rfModel = rfFittedModel.stages(0).asInstanceOf[RandomForestRegressionModel]
//    println("Learned regression forest model:\n" + rfModel.toDebugString)
//
//    // Train a GBT model.
//    val gbt = new GBTRegressor()
//      .setLabelCol("SalePrice")
//      .setFeaturesCol("features")
//      .setMaxIter(10)
//
//    // Chain indexer and GBT in a Pipeline.
//    val pipeline = new Pipeline()
//      .setStages(Array(gbt))
//
//    // Train model. This also runs the indexer.
//    val model = pipeline.fit(trainingData)
//
//    // Make predictions.
//    val predictions = model.transform(testData)
//
//    // Select example rows to display.
//    predictions.select("prediction", "SalePrice", "features").show(5)
//
//    // Select (prediction, true label) and compute test error.
//    val evaluator = new RegressionEvaluator()
//      .setLabelCol("SalePrice")
//      .setPredictionCol("prediction")
//      .setMetricName("rmse")
//    val rmse = evaluator.evaluate(predictions)
//    println("Root Mean Squared Error (RMSE) on GBT model test data = " + rmse)
//
//    val gbtModel = model.stages(0).asInstanceOf[GBTRegressionModel]
//
//
//
//    println("Learned regression GBT model:\n" + gbtModel.toDebugString)
//
//    println("Decision Tree RMSE = " + rmse2)
//    println("RandomForestRegressor RMSE = " + RFrmse)
//    println("Gradient-boosted Tree RMSE = " + rmse)

  }
}