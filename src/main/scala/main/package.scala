import org.apache.log4j.{Logger, Level}

import java.nio.file.{Paths, Files}

package object main {

  // Remove Spark's INFO logs
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  /** Parses standard input arguments */
  def parseArgs(args: Array[String]): Map[String, String] = {
    val usage = """
      Usage: [--input-path string] [--classifier-name string] [--output-folder string]
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
        case "--model-folder" :: value :: tail =>
          nextOption(map ++ Map("modelFolder" -> value), tail)
        case option :: tail =>
          println(s"Unknown option ${option}")
          sys.exit(1)
      }
    }

    // Set default values
    val defaultOptions = Map[String, String](
      "inputPath" -> "datasets/bosch/bosch-less.data",
      "classifierName" -> "DT",
      "modelFolder" -> "models"
    )
    return nextOption(defaultOptions, arglist)
  }

  /** Checks if a file exists in the given path */
  def fileExists(path: String): Boolean = {
    return Files.exists(Paths.get(path))
  }

}
