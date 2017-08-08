
import java.util.Calendar

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, GBTClassifier, LogisticRegression}
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
    val srcDataDir = System.getProperty("user.dir") + "/source-data/"

    val dataExtractor = new DataExtractor(srcDataDir, spark)

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

    // Select example rows to display.
    val outputFile = System.getProperty("user.dir") + "/housing-predictions/" + Calendar.getInstance().getTime.toString
    predictions2.withColumnRenamed("prediction", "SalePrice").select("Id","SalePrice").coalesce(1).write.option("header", "true").csv(outputFile)
    predictions2.show(50)
    println("Prediction output exported as " + outputFile + ".csv")
    spark.stop()
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