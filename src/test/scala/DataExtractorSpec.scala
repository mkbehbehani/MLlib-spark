import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.scalatest._

class DataExtractorSpec extends FunSpec with Matchers {
  val sparkSession = SparkSession
    .builder
    .master("local[*]")
    .appName("PipelineExample")
    .getOrCreate()
  val srcDataDir = System.getProperty("user.dir") + "/source-data/"
  describe("A DataExtractor"){
    describe("When given a spark session and source data directory data"){
      val dataExtractor = new DataExtractor(srcDataDir, sparkSession)
      it ("extracts csv training data to a DataFrame"){
        assert(dataExtractor.trainingData.isInstanceOf[sql.DataFrame])
      }
      it ("extracts csv test data to a DataFrame"){
        assert(dataExtractor.testData.isInstanceOf[sql.DataFrame])
      }
    }
  }
  sparkSession.stop()
}
