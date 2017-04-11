import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import java.util.Calendar
import org.apache.commons.io.FilenameUtils

object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val shakespeareFiles: RDD[(String, String)] = sc.wholeTextFiles(System.getProperty("user.dir") + "/hw3data")
    val termsAndFiles = shakespeareFiles.flatMap(_._2.split("\\r?\\n")).map(s =>(s.split(":::")(1).split("::")(0), s.toLowerCase.split(":::")(2).split("::")(0).split("\\W")))
      .map(t => (t._1, t._2.map((t._1, _)))).flatMap(_._2)
      .groupByKey()
      .map(t => (t._1, t._2.map((_,1))))
      .map(l=>(l._1, l._2.foldLeft(List[(String, Int)]())((accum, curr)=>{ // array reduction, from this stack overflow: http://stackoverflow.com/questions/30089646/spark-use-reducebykey-on-nested-structure
        val accumAsMap = accum.toMap
        accumAsMap.get(curr._1) match {
          case Some(value : Int) => (accumAsMap + (curr._1 -> (value + curr._2))).toList
          case None => curr :: accum
        }
      })))
     .map(t=>(t._1, t._2.mkString("[",",", "]"))) // Format the file output
    termsAndFiles.coalesce(1).saveAsTextFile(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString)
    sc.stop()
  }
}
