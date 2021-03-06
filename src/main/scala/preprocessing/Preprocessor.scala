package preprocessing

import utils._

import org.apache.spark.ml.feature.{
  QuantileDiscretizer,
  Bucketizer,
  PCA,
  PCAModel,
  VectorAssembler,
  StandardScaler,
  Imputer
}
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.linalg.{Vector, DenseMatrix}
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.functions.{
  trim,
  count,
  isnan,
  when,
  length,
  col,
  udf,
  lit,
  sum,
  countDistinct,
  approx_count_distinct
}

object Preprocessor {

  /** Takes a random subset of samples from the given DataFrame */
  def takeSubset(
      data: DataFrame,
      p: Double = 0.5
  ): DataFrame = {
    Logger.info(
      s"Retrieving a random subset of samples of dimension ${p * data.count}"
    )
    val Array(toReturn, toDrop) =
      data.randomSplit(Array(p, 1 - p), seed = Utils.seed)
    return toReturn
  }

  /** Counts the number of null values in the given Column */
  def countNulls(c: Column, nanAsNull: Boolean = false): Column = {
    val pred = c.isNull or (if (nanAsNull) isnan(c) else lit(false))
    return sum(pred.cast("integer"))
  }

  /** Removes columns where all values are the same,
    * including all null columns
    */
  def dropConstantColumns(
      data: DataFrame,
      approx: Boolean = true
  ): DataFrame = {
    Logger.info(s"Dropping null/constant columns")

    // Defines a lambda expression based on the value
    // of the `approx` parameter
    val count = (columnName: String) => {
      if (!approx) countDistinct(columnName)
      else approx_count_distinct(columnName)
    }

    // Computes a new row containing 0 for columns to drop
    // and 1 for columns to keep
    val row = data
      .select(
        data.columns.map(c =>
          when(count(c).isInCollection(Array(0, 1)), 0)
            .otherwise(1)
            .as(c)
        ): _*
      )
      .groupBy()
      .max()
      .head()

    // Selects columns to keep
    val toKeep = row
      .getValuesMap[Int](row.schema.fieldNames)
      .map { c => if (c._2 == 1) Some(c._1) else None }
      .flatten
      .toArray

    // Returns columns to keep (removing the max() substring)
    return data.select(
      row.schema.fieldNames
        .intersect(toKeep)
        .map(c => col(c.drop(4).dropRight(1))): _*
    )
  }

  /** Drops duplicated rows in the DataFrame */
  def dropDuplicates(data: DataFrame): DataFrame = {
    Logger.info(s"Dropping non-unique rows")
    return data.dropDuplicates()
  }

  /** Drops the given list of columns */
  def dropColumns(data: DataFrame, toDrop: String*): DataFrame = {
    Logger.info(s"Dropping columns: ${toDrop.mkString("[", ", ", "]")}")
    return data.drop(toDrop: _*)
  }

  /** Maintains only the given list of columns */
  def maintainColumns(data: DataFrame, toMaintain: Array[String]): DataFrame = {
    val toDrop = data.columns.filterNot(c => toMaintain.contains(c))
    return dropColumns(data, toDrop: _*)
  }

  /** Drops rows that contain at least one null value */
  def dropNullRows(data: DataFrame): DataFrame = {
    Logger.info(s"Dropping rows containing any null or NaN values")
    return data.na.drop
  }

  /** Apply the given functions over each and every DataFrame Column */
  def applyOverColumns(data: DataFrame, f: (Column) => Column): DataFrame = {
    return data.columns.foldLeft(data) { (df, c) =>
      df.withColumn(c, f(col(c)))
    }
  }

  /** Drops rows that contain the given value */
  def removeRowsWithValue(data: DataFrame, value: String): DataFrame = {
    Logger.info(
      s"""Dropping rows containing the value "${value}" in at least one column"""
    )
    return data.columns
      .map(c => {
        data.filter(col(c) !== value)
      })
      .reduceRight(_ intersect _)
  }

  /** Substitutes values matching the given one to null values in the DataFrame */
  def valuesToNull(data: DataFrame, value: String): DataFrame = {
    Logger.info(s"""Converting values "${value}" to null""")
    return applyOverColumns(
      data,
      c => when(c.equalTo(value), null).otherwise(c)
    )
  }

  /** Converts null values to the mean/median value of each column */
  def nullToValues(
      data: DataFrame,
      method: String = "mean",
      exclude: Array[String] = Array()
  ): DataFrame = {
    Logger.info(
      s"Applying ${method} feature imputation, excluding columns " +
        s"${exclude.mkString("[", ", ", "]")}"
    )
    val inputCols = data.columns.filterNot(c => exclude.contains(c))
    val imputer = new Imputer()
      .setInputCols(inputCols)
      .setOutputCols(inputCols)
      .setStrategy(method)

    imputer.fit(data).transform(data)
  }

  /** Trims column names and DataFrame values */
  def trimValues(data: DataFrame): DataFrame = {
    Logger.info(s"Trimming column names and DataFrame values")
    val dataTypes: Map[String, String] = data.dtypes.toMap
    var newData = data.columns.foldLeft(data) { (df, c) =>
      df.withColumnRenamed(c, c.replaceAll("\\s", ""))
    }
    return applyOverColumns(
      newData,
      c => {
        if (dataTypes(c.toString) == "string") trim(c)
        else c
      }
    )
  }

  /** Converts features into binary (1 = Present value / 0 = Missing value) */
  def binaryConversion(
      data: DataFrame,
      exclude: Array[String] = Array()
  ): DataFrame = {
    return applyOverColumns(
      data,
      c => {
        if (!exclude.contains(c.toString)) {
          Logger.info(s"""Converting column "${c}" to 0/1 values""")
          when(c.isNull, 0).otherwise(1)
        } else c
      }
    )
  }

  /** Performs Principal Component Analysis to reduce the number of features */
  def pca(
      inputData: DataFrame,
      maxComponents: Int,
      assembleFeatures: Boolean = true,
      standardizeFeatures: Boolean = true,
      explainedVariance: Double = 0.95,
      exclude: Array[String] = Array()
  ): Tuple2[DataFrame, DenseMatrix] = {
    require(
      explainedVariance > 0 && explainedVariance <= 1,
      "Invalid value for the explainedVariance parameter."
    )
    Logger.info(s"Performing Principal Components Analysis")

    // Assemble input features into a single vector
    val featuresCol = "features"
    val pcaFeaturesCol = "pcaFeatures"
    var data: DataFrame = inputData
    if (assembleFeatures)
      data = assemble(
        inputData,
        outputCol = featuresCol,
        inputCols = Some(data.columns.filterNot(c => exclude.contains(c)))
      )

    // Clip the maximum number of components
    val numFeatures = data.select(featuresCol).first.getAs[Vector](0).size
    val maxComp = Utils.clip(maxComponents, 1, numFeatures)

    // Standardize features to zero mean, unit variance
    if (standardizeFeatures) data = standardize(data, featuresCol)

    // Perform PCA
    val pca = new PCA()
      .setInputCol(featuresCol)
      .setOutputCol(pcaFeaturesCol)
      .setK(maxComp)
    val fittedModel = pca.fit(data)

    // Get the components accounting for the given explained variance
    val variances =
      Utils.take(fittedModel.explainedVariance.toArray, explainedVariance)
    val numComponents = variances.length match {
      case 0 => maxComp
      case _ => variances.length
    }
    Logger.info(
      s"The number of principal components explaining ${explainedVariance * 100}% of the variance is ${numComponents}"
    )

    // Apply transformation
    val numRows = fittedModel.pc.numRows
    val numCols = fittedModel.pc.numCols
    val values = fittedModel.pc.values.clone().slice(0, numRows * numComponents)
    val pc = new DenseMatrix(numRows, numComponents, values)
    val comp = toComponents(data, pc, featuresCol, exclude)
    return (
      vectorToDataFrame(comp, featuresCol, maintainVector = true),
      pc
    )
  }

  /** Converts a DataFrame with a vector column to an expanded DataFrame,
    * based on the given features column
    */
  def vectorToDataFrame(
      data: DataFrame,
      vectorCol: String,
      maintainVector: Boolean = false
  ): DataFrame = {
    Logger.info(
      s"""Converting vector column "${vectorCol}" to single columns"""
    )
    val size = data.select(vectorCol).first.getAs[Vector](0).size
    val exprs =
      (0 until size).map(i => col("_tmp_vec").getItem(i).alias(s"f$i"))
    val newData = data
      .select(vector_to_array(col(vectorCol)).alias("_tmp_vec"))
      .select(exprs: _*)
    val mergedData = Utils.mergeDataFrames(newData, data.select("*"))
    if (!maintainVector) return mergedData.drop(vectorCol)
    return mergedData
  }

  /** Transform features to principal components */
  def toComponents(
      data: DataFrame,
      pc: DenseMatrix,
      featuresCol: String,
      maintain: Array[String] = Array()
  ): DataFrame = {
    Logger.info(s"Transforming features to principal components")
    val transposed = pc.transpose
    val transformer = udf { vector: Vector => transposed.multiply(vector) }
    return data
      .withColumn(
        "_tmp",
        transformer(col(featuresCol))
      )
      .drop(featuresCol)
      .withColumnRenamed("_tmp", featuresCol)
      .select(featuresCol, maintain: _*)
  }

  /** Bins the given column values according to the defined splits */
  def binning(
      data: DataFrame,
      columnName: String,
      splits: Array[Double]
  ): DataFrame = {
    // format: off
    Logger.info(
      s"""Binning column "${columnName}" with splits ${splits.mkString("[", ", ", "]")}"""
    )
    // format: on

    val bucketizer = new Bucketizer()
      .setInputCol(columnName)
      .setOutputCol(s"T${columnName}")
      .setSplits(splits)

    return bucketizer
      .transform(data)
      .drop(columnName)
      .withColumnRenamed(s"T${columnName}", columnName)
  }

  /** Discretizes the given column values according to the specified number of quantiles */
  def quantileDiscretizer(
      data: DataFrame,
      columnName: String,
      numBuckets: Int
  ): DataFrame = {
    Logger.info(
      s"""Discretizing column "${columnName}" to ${numBuckets} quantiles"""
    )

    val discretizer = new QuantileDiscretizer()
      .setInputCol(columnName)
      .setOutputCol(s"T${columnName}")
      .setNumBuckets(numBuckets)

    return discretizer
      .fit(data)
      .transform(data)
      .drop(columnName)
      .withColumnRenamed(s"T${columnName}", columnName)
  }

  /** Assembles the given columns (or every column) into a single one */
  def assemble(
      data: DataFrame,
      outputCol: String,
      inputCols: Option[Array[String]] = None
  ): DataFrame = {
    inputCols match {
      case Some(value) => {
        // format: off
          Logger.info(
            s"""Assembling columns ${value.mkString("[", ", ", "]")} into column "${outputCol}""""
          )
          // format: on
        return new VectorAssembler()
          .setInputCols(value)
          .setOutputCol(outputCol)
          .transform(data)
      }
      case None => {
        Logger.info(s"""Assembling all columns into column "${outputCol}"""")
        return new VectorAssembler()
          .setInputCols(data.columns)
          .setOutputCol(outputCol)
          .transform(data)
      }
    }
  }

  /** Applies features standardization (zero mean and unit variance) */
  def standardize(
      data: DataFrame,
      inputCol: String,
      withMean: Boolean = true,
      withStd: Boolean = true
  ): DataFrame = {
    Logger.info(s"""Standardizing column "${inputCol}"""")
    return new StandardScaler()
      .setWithMean(withMean)
      .setWithStd(withStd)
      .setInputCol(inputCol)
      .setOutputCol(s"T${inputCol}")
      .fit(data)
      .transform(data)
      .drop(inputCol)
      .withColumnRenamed(s"T${inputCol}", inputCol)
  }

}
