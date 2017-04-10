import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import java.util.Calendar
import org.apache.commons.io.FilenameUtils



object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val shakespeareFiles: RDD[(String, String)] = sc.wholeTextFiles(System.getProperty("user.dir") + "/shakespeare")
    val termAndFiles = shakespeareFiles.map(tup => (FilenameUtils.getBaseName(tup._1), tup._2.split("\\W"))).map(tup => (tup._1, tup._2.map((_,tup._1)))).flatMap(_._2).distinct().groupByKey()
//      val historiesCount =
//      .map(tup => (tup._1, sc.parallelize(tup._2)).map(tup => (tup._1, tup._2.reduceByKey(_+_)))

      //    val counts2 = sc.makeRDD(counts.)
//    val supercount =  counts.map(tup => (tup._1, tup._2.PairRDDFunctions.reduceByKey({ case (x: String, y: Int) => x + y })))
//      .map(tup => (tup._1, tup._2.reduceByKey({ case (x: String, y: Int) => x + y })))
    //      val termInstances: Array[(String, Int)] = (fileCollection: org.apache.spark.rdd.RDD[(String, String)]) => fileCollection.map(_._2).flatMap(_.split("\\W")).map((_, 1)).reduceByKey({ case (x, y) => x + y })
    //    val reducedCounts = sc.parallelize(counts.map(tup => (tup._1, tup._2.reduceByKey({ case (x, y) => x + y }))))

    termAndFiles.coalesce(1).saveAsTextFile(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString)
    sc.stop()
  }
}
