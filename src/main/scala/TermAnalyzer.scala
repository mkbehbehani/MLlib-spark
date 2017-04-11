import java.util.Calendar
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val shakespeareFiles: RDD[(String, String)] = sc.wholeTextFiles(System.getProperty("user.dir") + "/hw3data")
    val stopWords = List("about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having",
    "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows",
    "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "its", "itself", "lets", "me",
    "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours ", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes",
    "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them",
    "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were",
    "weve", "were", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos",
    "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your",
    "yours", "yourself", "yourselves")

    val termsAndFiles = shakespeareFiles.flatMap(_._2.split("\\r?\\n")) // split each file's contents by line
      .map(s =>(s.split(":::")(1).split("::")(0), s.toLowerCase.split(":::")(2).split("::")(0).split("\\W").filter(_.length > 2).filter(!stopWords.contains(_)))) // extract (author, [word1, word2, word3]), words filtered for stopwords and length
      .map(t => (t._1, t._2.map((t._1, _)))).flatMap(_._2) // create (author, word) tuples for every word
      .groupByKey()  // Group word occurrences into a tuple of (author, (file1, file2, file3..))
      .map(t => (t._1, t._2.map((_,1))))  // create tuples of (filename, 1)
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
