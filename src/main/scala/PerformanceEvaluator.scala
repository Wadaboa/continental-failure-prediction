import preprocessing.{BoschDataset}
import prediction.{Predictor}

import org.apache.spark.sql.{SparkSession, DataFrame}
import preprocessing.Preprocessor

object PerformanceEvaluator {

  def main(args: Array[String]): Unit = {
    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val classifierName = options.get("classifierName")

    // Create SparkSession object
    val spark =
      SparkSession.builder.appName("Production line performance").getOrCreate()

    // Create Dataset object (and read data)
    val dataset = new BoschDataset(inputPath.orNull.toString)
    dataset.data.show()

    // Preprocess data
    val x = dataset.preprocess()
    x.show()

    /*
    // Train the classifier and test it
    val classifier = Classifier(classifierName.orNull.toString, dataset)
    classifier.train()
    val result = classifier.test()
    println(result)
     */

    // Stop SparkSession execution
    spark.stop()
  }

  def parseArgs(args: Array[String]): Map[String, String] = {
    val usage = """
      Usage: [--input-path string] [--classifier-name string]
    """
    if (args.length == 0) println(usage)
    val arglist = args.toList

    // Parse options
    def nextOption(
        map: Map[String, String],
        list: List[String]
    ): Map[String, String] = {
      list match {
        case Nil => map
        case "--input-path" :: value :: tail =>
          nextOption(map ++ Map("inputPath" -> value), tail)
        case "--classifier-name" :: value :: tail =>
          nextOption(map ++ Map("classifierName" -> value), tail)
        case option :: tail =>
          println(s"Unknown option ${option}")
          sys.exit(1)
      }
    }

    // Set default values
    val defaultOptions = Map[String, String](
      "inputPath" -> "dataset/adult.data",
      "classifierName" -> null
    )
    return nextOption(defaultOptions, arglist)
  }

}
