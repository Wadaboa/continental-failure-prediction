package preprocessing

import org.apache.spark.sql.{DataFrame}

object BoschDataset extends DatasetProperty {

  override def getDiscreteColumnNames(): Array[String] = {
    return Array()
  }

  override def getContinuosColumnNames(): Array[String] = {
    return Array()
  }

  override def getTargetColumnNames(): Array[String] = {
    return Array("Response")
  }

}

class BoschDataset(inputPath: String) extends Dataset(inputPath) {

  override def property = BoschDataset

  override def preprocess(): DataFrame = {
    val funcs = Seq(
      Preprocessor.trimValues(_: DataFrame),
      Preprocessor.maintainColumns(_: DataFrame, property.getColumnNames()),
      Preprocessor.valuesToNull(_: DataFrame, "?"),
      Preprocessor.valuesToNull(_: DataFrame, ""),
      Preprocessor.dropNullRows(_: DataFrame),
      Preprocessor.dropDuplicates(_: DataFrame),
      Preprocessor.binning(_: DataFrame, "age", Array(0, 18, 30, 60, 100)),
      Preprocessor
        .binning(_: DataFrame, "hours-per-week", Array(0, 25, 40, 60, 100)),
      Preprocessor.quantileDiscretizer(_: DataFrame, "capital-gain", 10),
      Preprocessor.quantileDiscretizer(_: DataFrame, "capital-loss", 5)
    )
    return funcs.foldLeft(data) { (r, f) => f(r) }
  }

}
