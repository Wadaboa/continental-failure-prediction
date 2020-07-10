package prediction

import scala.collection.mutable.HashMap

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, MultilayerPerceptronClassifier}


object Classifier {

  val classifiers = HashMap(
    "MLP" -> new MultilayerPerceptronClassifier()
              .setLayers(Array[Int](1, 2, 3))
              .setBlockSize(128)
              .setMaxIter(1000)
              .setSeed(42),
    "DT" -> new DecisionTreeClassifier()
              .setSeed(42)
  )

  def train(data: DataFrame, classifierName: String): Unit = {
    
    // Get the classifier from the list of available ones
    val classifier = classifiers.getOrElse(
      classifierName, 
      sys.error(s"Classifier not available. Please, select from ${classifiers.keySet}.")
    )

    // Index labels, adding metadata to the label column
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Put every feature into a single vector
    val vecAssembler = new VectorAssembler()
      .setInputCols(Array())
      .setOutputCol("features")
      .fit(data)

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2), seed=42)
    println(f"There are ${trainingData.count} rows in the training set, and ${testData.count} in the test set.")

    // Convert index labels back to original labels
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, vecAssembler, classifier, labelConverter))

    // Train model
    val model = pipeline.fit(trainingData)

    // Make predictions
    val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test error: ${(1.0 - accuracy)}")

  }
  
}
