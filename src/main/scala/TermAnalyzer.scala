/**
  * Created by Mashallah on 4/5/17.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD


object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val shakespeareFiles: RDD[(String, String)] = sc.wholeTextFiles("shakespeare")
    val counts = shakespeareFiles.map(_._2).flatMap(_.split("\\W")).map((_, 1)).reduceByKey { case (x, y) => x + y }
    counts.saveAsTextFile("wordcount4002")
    sc.stop()
  }
}