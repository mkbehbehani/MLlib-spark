import java.util.Calendar

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
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

    val df = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/source-data/train.csv").na.drop

    val fields = df.schema.fields filter {
      x => x.dataType match {
        case x: org.apache.spark.sql.types.StringType => true
        case _ => false
      }
    } map { x => x.name }

    val noStringDf = fields.foldLeft(df){ case(dframe,field) => dframe.drop(field) }

    val labelColumn = noStringDf.columns.reverse.head
    val featureColumns = noStringDf.columns.tail
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val featurizedData = assembler.transform(noStringDf)
    featurizedData.show

    val arrfields = df.schema.fields filter {
      x => x.dataType match {
        case x: org.apache.spark.sql.types.ArrayType => true
        case _ => false
      }
    } map { x => x.name }
    featurizedData
    val output = featurizedData.withColumnRenamed("SalePrice", "label")
    val classificationDF = output.select("label", "features")
    classificationDF.show

    val lr = new LinearRegression()

    // Fit the model
    val lrModel = lr.fit(classificationDF)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

  }
}
// scalastyle:on println