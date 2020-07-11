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

  def getPipeline(): Pipeline = {
    // Index labels, adding metadata to the label column
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Put every feature into a single vector
    val vecAssembler = new VectorAssembler()
      .setInputCols(Array())
      .setOutputCol("features")

    // Convert index labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, vecAssembler, model, labelConverter))
    
    return pipeline
  }

  def train(): Unit = {
    trainedModel = pipeline.fit(trainingData)
  }

  def test(): Double = {
    val predictions = trainedModel.transform(testData)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName(metricName)
    return evaluator.evaluate(predictions)
  }
  
}
