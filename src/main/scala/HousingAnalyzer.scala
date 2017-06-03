import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object HousingAnalyzer {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("PipelineExample")
      .getOrCreate()

    val rawTrainingData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/train.csv")
    val rawTestData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/test.csv")

    val countFeaturesTrainingDF = rawTrainingData.select("Id","SalePrice","MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    val nullCorrectedTrainingDf = countFeaturesTrainingDF.na.fill(countFeaturesTrainingDF.columns.zip(
      countFeaturesTrainingDF.select(countFeaturesTrainingDF.columns.map(mean): _*).first.toSeq
    ).toMap)

    val countFeaturesTestDF = rawTestData.select("Id","MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    val nullCorrectedTestDf = countFeaturesTestDF.na.fill(countFeaturesTestDF.columns.zip(
      countFeaturesTestDF.select(countFeaturesTestDF.columns.map(mean): _*).first.toSeq
    ).toMap)
    val featureColumns = Array("LotArea","1stFlrSF","2ndFlrSF")
    val labelColumn = "SalePrice"

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val processedTrainingData = assembler.transform(nullCorrectedTrainingDf).select("Id","SalePrice", "features")
    val processedTestData = assembler.transform(nullCorrectedTestDf).select("Id","features")
    processedTrainingData.show(3)
    processedTestData.show(3)

    val trainingData = processedTrainingData
    val testData = processedTestData
//    val Array(trainingData, testData) = processedTrainingData.randomSplit(Array(0.8, 0.2))

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
    predictions2.show(100)

    // Select (prediction, true label) and compute test error.
    val evaluator2 = new RegressionEvaluator()
      .setLabelCol("SalePrice")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse2 = evaluator2.evaluate(predictions2)

    val treeModel = model2.stages(0).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)

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
    println("Decision Tree RMSE = " + rmse2)
//    println("RandomForestRegressor RMSE = " + RFrmse)
//    println("Gradient-boosted Tree RMSE = " + rmse)

  }
}
// scalastyle:on println