import org.apache.spark.sql.SparkSession

package object main {

  // Create shared SparkSession object
  val spark =
    SparkSession.builder.appName("Production line performance").getOrCreate()

  /** Parses standard input arguments */
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
      "inputPath" -> "datasets/bosch/bosch-less.data",
      "classifierName" -> "DT"
    )
    return nextOption(defaultOptions, arglist)
  }

  /** Stops Spark execution */
  def stopSpark(): Unit = spark.stop()

}
