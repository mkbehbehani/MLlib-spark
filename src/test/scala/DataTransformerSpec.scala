import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.scalatest._

class DataTransformerSpec extends FunSpec with Matchers {
  val sparkSession = SparkSession
    .builder
    .master("local[*]")
    .appName("PipelineExample")
    .getOrCreate()
  val srcDataDir = System.getProperty("user.dir") + "/source-data/"
  describe("A DataTransformer"){
    describe("When given a spark session and source data directory data"){
      val dataTransformer = new DataTransformer(trainingData, testData, featureColumns, predictionColumn)
      it ("transforms the training DataFrame into id, feature and prediction columns"){
        assert(dataTransformer.trainingData.columns == Array("Id", featureColumns, predictionColumn))
      }
      it ("transforms the test DataFrame into id and feature columns"){
        assert(dataTransformer.trainingData.columns == Array("Id", featureColumns))
      }
    }
  }
  sparkSession.stop()
}
