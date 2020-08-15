package prediction

import preprocessing.{Dataset}
import utils._

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.evaluation.{
  Evaluator,
  MulticlassClassificationEvaluator
}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.classification.{RandomForestClassifier => RF}

case class RandomForestClassifier(dataset: Dataset)
    extends Predictor[RF](dataset) {

  var impurity: String = "entropy"
  var numTrees: Int = 100
  var maxDepth: Int = 25
  var maxBins: Int = 64
  var minSamplesLeaf: Int = 10

  var model = new RF()
    .setImpurity(impurity)
    .setNumTrees(numTrees)
    .setMaxDepth(maxDepth)
    .setMaxBins(maxBins)
    .setMinInstancesPerNode(minSamplesLeaf)
    .setSeed(Utils.seed)
    .setLabelCol(labelCol)
    .setFeaturesCol(featuresCol)
    .setPredictionCol(predictionCol)

  override def paramGrid: Array[ParamMap] = {
    return new ParamGridBuilder()
      .addGrid(model.maxDepth, (15 to maxDepth by 5).toArray)
      .addGrid(model.minInstancesPerNode, (5 to minSamplesLeaf by 2).toArray)
      .build()
  }

  override def defaultEvaluator: Evaluator = {
    return new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predictionCol)
      .setMetricName(metricName)
  }

}
