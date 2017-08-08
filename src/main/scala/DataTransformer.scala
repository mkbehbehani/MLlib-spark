import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

class DataTransformer(trainingData, testData, featureColumns, predictionColumn) {

  val assembler = new VectorAssembler()
    .setInputCols(featureColumns)
    .setOutputCol("features")

  val featurizedTrainingData = assembler.transform(trainingData).select("Id", predictionColumn, "features")
  val featurizedTestData = assembler.transform(testData).select("Id","features")

}