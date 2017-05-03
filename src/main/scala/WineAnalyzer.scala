import java.util.Calendar

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor, LinearRegression}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.max

object WineAnalyzer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("PipelineExample")
      .getOrCreate()

    val rawTrainingData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/train.csv").na.drop
    val rawTestData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/test.csv").na.drop

    val fields = rawTrainingData.schema.fields filter {
      x => x.dataType match {
        case x: org.apache.spark.sql.types.StringType => true
        case _ => false
      }
    } map { x => x.name }

    val noStringTrainingDf = fields.foldLeft(rawTrainingData){ case(dframe,field) => dframe.drop(field) }

    val labelColumn = noStringTrainingDf.columns.reverse.head
    val featureColumns = noStringTrainingDf.columns.tail
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val featurizedTrainingData = assembler.transform(noStringTrainingDf)
    featurizedTrainingData.show

    val arrfields = rawTrainingData.schema.fields filter {
      x => x.dataType match {
        case x: org.apache.spark.sql.types.ArrayType => true
        case _ => false
      }
    } map { x => x.name }
    featurizedTrainingData
    val output = featurizedTrainingData.withColumnRenamed("SalePrice", "label")
    val processedTrainingData = output.select("label", "features")
    processedTrainingData.show

     //  test data processing

//    val testData = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/test.csv").na.drop
//
//    val testfields = rawTestData.schema.fields filter {
//      x => x.dataType match {
//        case x: org.apache.spark.sql.types.StringType => true
//        case _ => false
//      }
//    } map { x => x.name }
//
//    val noStringTestDf = testfields.foldLeft(rawTestData){ case(dframe,field) => dframe.drop(field) }
//
//    val testlabelColumn = noStringTestDf.columns.reverse.head
//    val testfeatureColumns = noStringTestDf.columns.tail
//    val testassembler = new VectorAssembler()
//      .setInputCols(testfeatureColumns)
//      .setOutputCol("features")
//
//    val featurizedTestData = testassembler.transform(noStringTestDf)
//    featurizedTestData.show
//
//    val testarrfields = rawTestData.schema.fields filter {
//      x => x.dataType match {
//        case x: org.apache.spark.sql.types.ArrayType => true
//        case _ => false
//      }
//    } map { x => x.name }
//    featurizedTestData
//    val testoutput = featurizedTestData.withColumnRenamed("SalePrice", "label")
//    val processedTestData = testoutput.select("label", "features")
//    processedTestData.show


    val Array(trainingSplitData1, trainingSplitData2) = processedTrainingData.randomSplit(Array(0.7, 0.3))

    // model building
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(trainingSplitData1)

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))


    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingSplitData1)

    // Make predictions.
    val predictions = model.transform(trainingSplitData2)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(100)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
  }
}
// scalastyle:on println