package prediction

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}


abstract class Classifier(data: DataFrame) {

  val model: PipelineStage
  val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed=getRandomSeed())
  val pipeline = getPipeline()
  var trainedModel: PipelineModel = _
  val metricName: String = "accuracy"

  /** Defines model-specific data transformations */
  def getPipeline(): Pipeline

  /** Trains the model */
  def train(): Unit = {
    trainedModel = pipeline.fit(trainingData)
  }

  /** Tests the model */
  def test(): Double = {
    val predictions = trainedModel.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName(metricName)
    return evaluator.evaluate(predictions)
  }
  
}
