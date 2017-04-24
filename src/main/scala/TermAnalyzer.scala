import java.util.Calendar

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types._

object TermAnalyzer {
  def main(args: Array[String]) {
    val sparkSession = SparkSession.builder().master("local[*]").getOrCreate()

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

    val df = sparkSession.read.format("com.databricks.spark.csv").schema(customSchema).option("delimiter", ";").option("header", "true").option("inferSchema", "true").load("/home/mashallah/IdeaProjects/spark-machine-learning-scala/hwdata/winequality-white.csv")


    val targetInd = df.columns.indexOf("quality")
    val ignored = List("quality")
    val featInd = df.columns.diff(ignored).map(df.columns.indexOf(_))
    val wineForML = df.rdd.map(r => LabeledPoint(
      r.getDouble(targetInd), // Get target value
      // Map feature indices to values
      Vectors.dense(featInd)
    ))
    wineForML.take(4)

    val wineSplit = df.randomSplit(Array(0.7, 0.3))
    val trainingData = wineSplit(0)
    val testing = wineSplit(1)

    wineForML.coalesce(1).saveAsTextFile(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString)


    sparkSession.stop()
  }
}
