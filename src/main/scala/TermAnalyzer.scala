/**
  * Created by Mashallah on 4/5/17.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD


object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val shakespeareFiles: RDD[(String, String)] = sc.wholeTextFiles("/Users/Mashallah/spark-scala-term-analyzer/shakespeare")
    val counts = shakespeareFiles.map(_._2).flatMap(_.split("\\W")).map((_, 1)).reduceByKey({ case (x, y) => x + y })
    counts.coalesce(1,true).saveAsTextFile("/Users/Mashallah/spark-scala-term-analyzer/spark-output/2")
    sc.stop()
  }
}