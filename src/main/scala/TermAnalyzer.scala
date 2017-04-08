import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import java.util.Calendar


object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val shakespeareFiles: RDD[(String, String)] = sc.wholeTextFiles(System.getProperty("user.dir") + "/shakespeare")
    val counts = shakespeareFiles.map(_._2).flatMap(_.split("\\W")).map((_, 1)).reduceByKey({ case (x, y) => x + y })
    counts.coalesce(1,true).saveAsTextFile(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString)
    sc.stop()
  }
}
