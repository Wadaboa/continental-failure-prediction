package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
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
  var numTrees: Int = 10
  var maxDepth: Int = 10
  var maxBins: Int = 64
  var minSamplesLeaf: Int = 3

  var model = new RF()
    .setImpurity(impurity)
    .setNumTrees(numTrees)
    .setMaxDepth(maxDepth)
    .setMaxBins(maxBins)
    .setSeed(getRandomSeed())
    .setLabelCol(labelCol)
    .setFeaturesCol(featuresCol)
    .setPredictionCol(predictionCol)

  override def paramGrid: Array[ParamMap] = {
    return new ParamGridBuilder()
      .addGrid(model.numTrees, (1 to numTrees).toArray)
      .addGrid(model.maxDepth, (1 to maxDepth).toArray)
      .addGrid(
        model.maxBins,
        (dataset.maxDistinctValues to maxBins by 10).toArray
      )
      .addGrid(model.minInstancesPerNode, (1 to minSamplesLeaf by 2).toArray)
      .build()
  }

  override def evaluator: Evaluator = {
    return new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predictionCol)
      .setMetricName(metricName)
  }

}
