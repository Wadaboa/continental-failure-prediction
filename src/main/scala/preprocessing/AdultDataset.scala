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

  override def getSchemaString(): Option[String] = {
    return Some("""
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
    """)
  }

}

class AdultDataset(
    inputPath: Option[String] = None,
    inputData: Option[DataFrame] = None
) extends Dataset(inputPath, inputData) {

  override type T = AdultDataset

  override def property = AdultDataset

  override def preprocess(): AdultDataset = {
    val funcs = Seq(
      Preprocessor.trimValues(_: DataFrame),
      Preprocessor
        .maintainColumns(_: DataFrame, property.getUsefulColumnNames()),
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
    return new AdultDataset(inputData = Some(funcs.foldLeft(data) { (r, f) =>
      f(r)
    }))
  }

}
