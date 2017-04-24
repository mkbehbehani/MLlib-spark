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
    val wineData = sparkSession.sparkContext.textFile("/home/mashallah/IdeaProjects/spark-machine-learning-scala/hwdata/winequality-white.csv")
    val parsedWineData = wineData.flatMap(_.split("\\n")).map(_.split(";")).map( _.map{y => y.toDouble}).map(y => LabeledPoint(y.last, Vectors.dense(y.slice(0, y.length-1))))

    val splits = parsedWineData.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))


    trainingData.coalesce(1).saveAsTextFile(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString)


    sparkSession.stop()
  }
}
