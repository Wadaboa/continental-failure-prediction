package preprocessing

import org.apache.spark.sql.{DataFrame}

object AdultDataset extends DatasetProperty {

  override def getDiscreteColumnNames(): Array[String] = {
    return Array(
      "age",
      "workclass",
      "education",
      "marital-status",
      "occupation",
      "relationship",
      "race",
      "sex",
      "hours-per-week",
      "native-country"
    )
  }

  override def getContinuosColumnNames(): Array[String] = {
    return Array(
      "capital-gain",
      "capital-loss"
    )
  }

  override def getTargetColumnNames(): Array[String] = {
    return Array("high-income")
  }

  override def getColumnNames(): Array[String] = {
    return getDiscreteColumnNames() ++ getContinuosColumnNames() ++ getTargetColumnNames()
  }

  override def getSchemaString(): String = {
    return """
      `age` INT,
      `workclass` STRING,
      `fnlwgt` INT,
      `education` STRING,
      `education-num` INT,
      `marital-status` STRING,
      `occupation` STRING,
      `relationship` STRING,
      `race` STRING,
      `sex` STRING,
      `capital-gain` INT,
      `capital-loss` INT,
      `hours-per-week` INT,
      `native-country` STRING,
      `high-income` STRING
    """
  }

}

class AdultDataset(data: DataFrame) extends Dataset(data) {

  import AdultDataset._

  if (!data.schema.equals(getSchema()))
    throw new IllegalArgumentException(
      "The given DataFrame does not respect the defined schema."
    )
  preprocess()

  override def preprocess(): DataFrame = {
    val funcs = Seq(
      Preprocessor.trimValues(_: DataFrame),
      Preprocessor.maintainColumns(_: DataFrame, getColumnNames()),
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
