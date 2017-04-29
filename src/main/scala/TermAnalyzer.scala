import java.util.Calendar

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.max

object TermAnalyzer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("PipelineExample")
      .getOrCreate()

    val customSchema = StructType(Array(
      StructField("fixed acidity", DoubleType, true),
      StructField("volatile acidity", DoubleType, true),
      StructField("citric acid", DoubleType, true),
      StructField("residual sugar", DoubleType, true),
      StructField("chlorides", DoubleType, true),
      StructField("free sulfur dioxide", DoubleType, true),
      StructField("total sulfur dioxide", DoubleType, true),
      StructField("density", DoubleType, true),
      StructField("pH", DoubleType, true),
      StructField("sulphates", DoubleType, true),
      StructField("alcohol", DoubleType, true),
      StructField("quality", DoubleType, true)
    ))

    val df = spark.read.format("com.databricks.spark.csv").schema(customSchema).option("delimiter", ";").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/MLlib-spark/hwdata/winequality-white.csv")
    import spark.implicits._


    val training = spark.read.format("libsvm").load("/home/mashallah/IdeaProjects/MLlib-spark/hwdata/sample_libsvm_data.txt")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // We can also use the multinomial family for binary classification
    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")

    val mlrModel = mlr.fit(training)

    // Print the coefficients and intercepts for logistic regression with multinomial family
    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

    // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
    // example
    val trainingSummary = lrModel.summary

    // Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    // Obtain the metrics useful to judge performance on test data.
    // We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
    // binary classification problem.
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    val roc = binarySummary.roc
    roc.show()
    roc.coalesce(1).write.option("header", "true").csv(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString + "/sample_file.csv")
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    // Set the model threshold to maximize F-Measure
    val fMeasure = binarySummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
      .select("threshold").head().getDouble(0)
    lrModel.setThreshold(bestThreshold)
    val resultOutput = s"Coefficients: ${lrModel.coefficients} " +
                       s"Intercept: ${lrModel.intercept}"


  }
}
// scalastyle:on println