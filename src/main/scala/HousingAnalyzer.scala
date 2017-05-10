import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.sql.SparkSession

object HousingAnalyzer {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("PipelineExample")
      .getOrCreate()

    val rawTrainingData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/train.csv").na.drop
    val rawTestData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/test.csv").na.drop

    val noStringTrainingDf = rawTrainingData.select("SalePrice","MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    val featureColumns = Array("MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    val labelColumn = "SalePrice"

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val featurizedTrainingData = assembler.transform(noStringTrainingDf)

    val output = featurizedTrainingData.withColumnRenamed("SalePrice", "label")
    val processedTrainingData = output.select("label", "features")

    val Array(trainingSplitData1, trainingSplitData2) = processedTrainingData.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(dt))


    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingSplitData1)

    // Make predictions.
    val predictions = model.transform(trainingSplitData2)

    // Select example rows to display.
    predictions.show(100)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val treeModel = model.stages(0).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
  }
}
// scalastyle:on println