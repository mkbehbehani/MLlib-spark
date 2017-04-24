import java.util.Calendar

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object TermAnalyzer {
  def main(args: Array[String]) {
    val sparkSession = SparkSession.builder().master("local[*]").getOrCreate()
    val df = sparkSession.read
      .format("com.databricks.spark.csv")
      .option("delimiter", ";")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("hwdata/winequality-white.csv")
    df.show
    sparkSession.stop()
  }
}
