import preprocessing.Preprocessor
import prediction.{Classifier, DecisionTreeClassifier}
import org.apache.spark.sql.{SparkSession, DataFrame}

object SimpleApp {

  def main(args: Array[String]): Unit = {

    // Parse command-line options
    val options = parseArgs(args)
    val inputPath = options.get("inputPath")
    val classifierName = options.get("classifierName")

    // Create SparkSession object
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

    // Read DataFrame
    val data = spark.read
      .schema(Preprocessor.getSchema())
      .format("csv")
      .option("delimiter", ",")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load(inputPath.orNull.toString)

    data.show()
    println(data.printSchema())
    var newData = Preprocessor.preprocess(data)
    println(newData.printSchema())
    newData.show()

    // Train each classifier
    /*
    classifierName match {
      case Some(value) => Classifier.train(data, value)
      case _ => sys.exit(1)
    }
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
    def nextOption(map: Map[String, String], list: List[String]): Map[String, String] = {
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
