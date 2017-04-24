import java.util.Calendar
import java.io._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object TermAnalyzer {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val wineData = sc.textFile("/home/mashallah/IdeaProjects/spark-machine-learning-scala/hwdata/winequality-white.csv")
    val parsedWineData = wineData.flatMap(_.split("\\n")).map(_.split(";")).map( _.map{y => y.toDouble}).map(y => LabeledPoint(y.last, Vectors.dense(y.slice(0, y.length-1))))
    val splits = parsedWineData.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))


    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
      maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
    val mseResults = "Test Mean Squared Error = " + testMSE
    val regressionModelResults = "Learned regression tree model:\n" + model.toDebugString

    val resultOutput:String = mseResults + regressionModelResults
    val results = sc.parallelize(resultOutput)

    // Save and load model
    model.save(sc, "target/tmp/" + Calendar.getInstance().getTime.toString + "myDecisionTreeRegressionModel")

    val pw = new PrintWriter(new File(System.getProperty("user.dir") + "/spark-output/" + Calendar.getInstance().getTime.toString))
    pw.write(resultOutput)
    pw.close
    sc.stop()
  }

}
