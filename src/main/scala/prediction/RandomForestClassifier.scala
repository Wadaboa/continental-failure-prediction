package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{
  Evaluator,
  MulticlassClassificationEvaluator
}
import org.apache.spark.ml.classification.{RandomForestClassifier => RF}

class RandomForestClassifier(dataset: Dataset) extends Predictor(dataset) {

  var impurity: String = "entropy"
  var numTrees: Int = 10
  var maxDepth: Int = 10
  var maxBins: Int = 64

  override type T = RF

  override def getModel(): RF = {
    return new RF()
      .setImpurity(impurity)
      .setNumTrees(numTrees)
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)
      .setSeed(getRandomSeed())
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predictionCol)
  }

  override def getEvaluator(): Evaluator = {
    return new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predictionCol)
      .setMetricName(metricName)
  }

}
