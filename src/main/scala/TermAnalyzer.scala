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
    val termsAndFiles = shakespeareFiles
                       .map(tup => (FilenameUtils.getBaseName(tup._1), tup._2.split("\\W"))) // For each file, extract the filename and split the entire textfile into a collection of individual words
                       .map(tup => (tup._1, tup._2.map((_,tup._1)))) // For every word in the file, create a (word, filename) tuple
                       .flatMap(_._2) // Discard the filename key, leaving us with (word, filename) tuples
                       .map(tup => (tup._1.toLowerCase, tup._2)) // Convert all words to lowercase
                       .distinct()  // Deduplicate file occurrences
                       .groupByKey() // Group filename occurrences into a tuple of (word, (file1, file2, file3..))
                       .map(tup=>(tup._1, tup._2.mkString("[",",", "]"))) // Format the file output
    termsAndFiles.coalesce(1).saveAsTextFile(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString)
    sc.stop()
  }
}
