import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.sql.SparkSession

object HousingAnalyzer {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.INFO)

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("PipelineExample")
      .getOrCreate()

    val featureSchema = StructType(Array(
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
     //  test data processing

    val testData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/test.csv").na.drop

    val testfields = rawTestData.schema.fields filter {
      x => x.dataType match {
        case x: org.apache.spark.sql.types.StringType => true
        case _ => false
      }
    } map { x => x.name }

    val noStringTestDf = testfields.foldLeft(rawTestData){ case(dframe,field) => dframe.drop(field) }

    val testlabelColumn = noStringTestDf.columns.reverse.head
    val testfeatureColumns = noStringTestDf.columns.tail
    val testassembler = new VectorAssembler()
      .setInputCols(testfeatureColumns)
      .setOutputCol("features")

    val featurizedTestData = testassembler.transform(noStringTestDf)

    val testarrfields = rawTestData.schema.fields filter {
      x => x.dataType match {
        case x: org.apache.spark.sql.types.ArrayType => true
        case _ => false
      }
    } map { x => x.name }
    featurizedTestData.show
    val processedTestData = featurizedTestData.select("Id","features").withColumnRenamed("Id", "label").limit(1)
    processedTestData.show

//    val Array(trainingSplitData1, trainingSplitData2) = processedTrainingData.randomSplit(Array(0.7, 0.3))
//
//    // model building
//    val featureIndexer = new VectorIndexer()
//      .setInputCol("features")
//      .setOutputCol("indexedFeatures")
//      .setMaxCategories(4)
//      .fit(processedTrainingData)
//
//    // Train a DecisionTree model.
//    val dt = new RandomForestRegressor()
//      .setLabelCol("label")
//      .setFeaturesCol("indexedFeatures")
//
//    // Chain indexer and tree in a Pipeline.
//    val pipeline = new Pipeline()
//      .setStages(Array(featureIndexer, dt))
//
//
//    // Train model. This also runs the indexer.
//    val model = pipeline.fit(trainingSplitData1)
//
//    processedTestData.rdd.saveAsTextFile(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString)
//    // Make predictions.
//    val Array(testSplitData1, testSplitData2) = processedTestData.randomSplit(Array(0.1, 0.9))
//
//    testSplitData1.show
//    val predictions = model.transform(testSplitData1)
//
//    // Select example rows to display.
//    predictions.show
//
//    // Select (prediction, true label) and compute test error.
//    val evaluator = new RegressionEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("rmse")
//    val rmse = evaluator.evaluate(predictions)
//    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
//
//    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
//    println("Learned regression tree model:\n" + treeModel.toDebugString)

    val Array(trainingSplitData1, trainingSplitData2) = processedTrainingData.randomSplit(Array(0.7, 0.3))

    // model building
//    val featureIndexer = new VectorIndexer()
//      .setInputCol("features")
//      .setOutputCol("indexedFeatures")
//      .setMaxCategories(4)
//      .fit(trainingSplitData1)

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
    val predictions = model.transform(processedTrainingData)



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