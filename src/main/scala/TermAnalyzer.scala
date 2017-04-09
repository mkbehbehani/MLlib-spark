/**
  * Created by Mashallah on 4/5/17.
  */

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._


object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val shakespeareFiles: RDD[(String, String)] = sc.wholeTextFiles("/Users/Mashallah/spark-scala-term-analyzer/shakespeare")
    val counts = shakespeareFiles.map(filenameContentTuple => (filenameContentTuple._1, filenameContentTuple._2)).map(termTuple => (termTuple._1, termTuple._2.split("\\W"))).map(tup => (tup._1, tup._2.map((_,1))))

      .map(tup => (tup._1, tup._2.reduceByKey({ case (x, y) => x + y })))
//      val termInstances: Array[(String, Int)] = (fileCollection: org.apache.spark.rdd.RDD[(String, String)]) => fileCollection.map(_._2).flatMap(_.split("\\W")).map((_, 1)).reduceByKey({ case (x, y) => x + y })
//    val reducedCounts = sc.parallelize(counts.map(tup => (tup._1, tup._2.reduceByKey({ case (x, y) => x + y }))))
    counts.coalesce(1,true).saveAsTextFile("/Users/Mashallah/spark-scala-term-analyzer/spark-output/3")
    sc.stop()
  }
}