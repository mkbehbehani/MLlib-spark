import java.util.Calendar

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
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

    val assembler = new VectorAssembler()
      .setInputCols(Array("fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"))
      .setOutputCol("features")

    val featurizedData = assembler.transform(df)
    val output = featurizedData.withColumnRenamed("quality", "label")
    output.show

    val training = spark.read.format("libsvm").load("/home/mashallah/IdeaProjects/MLlib-spark/hwdata/sample_libsvm_data.txt")


    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(output)

    // Print the coefficients and intercept for multinomial logistic regression
    println(s"Coefficients: \n${lrModel.coefficientMatrix}")
    println(s"Intercepts: ${lrModel.interceptVector}")

  }
}
// scalastyle:on println