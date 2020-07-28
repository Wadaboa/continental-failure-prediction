package preprocessing

import utils._

import org.apache.spark.sql.{DataFrame}

object ArrestDataset extends DatasetProperty {

  override def getContinuosColumnNames(): Array[String] = {
    return Array(
      "murder",
      "assault",
      "urban-pop",
      "rape"
    )
  }

  override def getSchemaString(): Option[String] = {
    return Some("""
      `city` STRING,
      `murder` DOUBLE,
      `assault` DOUBLE,
      `urban-pop` DOUBLE,
      `rape` DOUBLE
    """)
  }

}

case class ArrestDataset(
    inputPath: Option[String] = None,
    inputData: Option[DataFrame] = None
) extends Dataset(inputPath, inputData) {

  override type T = ArrestDataset

  override def property = ArrestDataset

  /** Assembles and standardizes features */
  def preprocess(): ArrestDataset = {
    val funcs = Seq(
      Preprocessor.assemble(
        _: DataFrame,
        outputCol = "features",
        inputCols = Some(data.columns.filterNot(c => c == "city"))
      ),
      Preprocessor.standardize(_: DataFrame, inputCol = "features")
    )
    return new ArrestDataset(inputData = Some(funcs.foldLeft(data) { (r, f) =>
      f(r)
    }))
  }

}
