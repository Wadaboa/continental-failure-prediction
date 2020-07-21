package prediction

import preprocessing.{Dataset}

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.evaluation.{
  Evaluator,
  MulticlassClassificationEvaluator
}
import org.apache.spark.ml.classification.{DecisionTreeClassifier => DT}

class DecisionTreeClassifier(dataset: Dataset) extends Predictor[DT](dataset) {

  override val metricName: String = "accuracy"

  override def getModel(): DT = {
    return new DT()
      .setImpurity("entropy")
      .setMaxDepth(10)
      .setSeed(getRandomSeed())
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxBins(64)
  }

  override def getPipeline(): Pipeline = {
    // Get column and target names
    val target = dataset.property.getTargetColumnNames()(0)
    val cols = dataset.getColumnNames().filter(c => c != target)

    // Index columns
    val columnsIndexer = new StringIndexer()
      .setInputCols(cols)
      .setOutputCols(cols.map("indexed-" + _))
      .fit(dataset.data)

    // Index labels
    val labelIndexer = new StringIndexer()
      .setInputCol(target)
      .setOutputCol("label")
      .fit(dataset.data)

    // Put every feature into a single vector
    val featuresAssembler = new VectorAssembler()
      .setInputCols(columnsIndexer.getOutputCols)
      .setOutputCol("features")

    // Convert index labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol(model.getPredictionCol)
      .setOutputCol("predicted-label")
      .setLabels(labelIndexer.labels)

    // Define the pipeline
    val pipeline = new Pipeline()
      .setStages(
        Array(
          columnsIndexer,
          labelIndexer,
          featuresAssembler,
          model,
          labelConverter
        )
      )

    return pipeline
  }

  override def getEvaluator(): Evaluator = {
    return new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName(metricName)
  }

}
