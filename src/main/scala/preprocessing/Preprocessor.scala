package preprocessing

import utils._

import org.apache.spark.ml.feature.{
  QuantileDiscretizer,
  Bucketizer,
  PCA,
  PCAModel,
  VectorAssembler,
  StandardScaler
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
  countDistinct
}

object Preprocessor {

  /** Takes a random subset of samples from the given DataFrame */
  def takeSubset(
      data: DataFrame,
      percentage: Option[Double] = None
  ): DataFrame = {
    var p: Double = 1.0
    percentage match {
      case Some(value) => p = value
      case None        => p = math.random()
    }
    val Array(toReturn, toDrop) =
      data.randomSplit(Array(p, 1 - p), seed = Utils.seed)
    return toReturn
  }

  /** Converts a DataFrame with a vector column to an expanded DataFrame,
    * based on the given features column
    */
  def fromVectorToDataframe(
      data: DataFrame,
      vectorCol: String,
      maintainVector: Boolean = false
  ): DataFrame = {
    val newData = data.select(vector_to_array(col(vectorCol)).alias("_tmp"))
    if (maintainVector) return newData.withColumn(vectorCol, data(vectorCol))
    return newData
  }

  /** Counts the number of null values in the given Column */
  def countNulls(c: Column, nanAsNull: Boolean = false): Column = {
    val pred = c.isNull and (if (nanAsNull) isnan(c) else lit(true))
    return sum(pred.cast("integer"))
  }

  /** Removes columns with all null values */
  def dropNullColumns(data: DataFrame): DataFrame = {
    val counts = data.columns
      .map(c =>
        (c, data.agg(countNulls(col(c), nanAsNull = true)).first.getLong(0))
      )
    val toDrop = counts.filter(x => x._2 == 1).map(_._1)
    Logger.info(s"Dropping null columns ${toDrop.mkString(" ")}")
    return dropColumns(data, toDrop: _*)
  }

  /** Removes columns where all values are the same */
  def dropConstantColumns(data: DataFrame): DataFrame = {
    val counts = data.columns
      .map(c => (c, data.agg(countDistinct(c)).first.getLong(0)))
    val toDrop = counts.filter(x => x._2 == 1).map(_._1)
    Logger.info(s"Dropping constant columns ${toDrop.mkString(" ")}")
    return dropColumns(data, toDrop: _*)
  }

  /** Drops duplicated rows in the DataFrame */
  def dropDuplicates(data: DataFrame): DataFrame = {
    return data.dropDuplicates()
  }

  /** Drops the given list of columns */
  def dropColumns(data: DataFrame, toDrop: String*): DataFrame = {
    return data.drop(toDrop: _*)
  }

  /** Maintains only the given list of columns */
  def maintainColumns(data: DataFrame, toMaintain: Array[String]): DataFrame = {
    val toDrop = data.columns.filterNot(c => toMaintain.contains(c))
    Logger.info(s"Dropping columns ${toDrop.mkString(" ")}")
    return dropColumns(data, toDrop: _*)
  }

  /** Drops rows that contain at least one null value */
  def dropNullRows(data: DataFrame): DataFrame = {
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
    val dfs = data.columns.map(c => {
      data.filter(col(c) !== value)
    })
    return dfs.reduceRight(_ intersect _)
  }

  /** Substitutes values matching the given one to null values in the DataFrame */
  def valuesToNull(data: DataFrame, value: String): DataFrame = {
    return applyOverColumns(
      data,
      c => when(c.equalTo(value), null).otherwise(c)
    )
  }

  /** Trims column names and DataFrame values */
  def trimValues(data: DataFrame): DataFrame = {
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
  def binaryConversion(data: DataFrame): DataFrame = {
    return applyOverColumns(
      data,
      c => when(c.isNull, 1).otherwise(0)
    )
  }

  /** Perform Principal Component Analysis to reduce the number of features */
  def pca(
      data: DataFrame,
      maxComponents: Int,
      assembleFeatures: Boolean = true,
      explainedVariance: Double = 0.95
  ): DataFrame = {
    require(
      explainedVariance > 0 && explainedVariance <= 1,
      "Invalid value for the explainedVariance parameter."
    )
    require(maxComponents > 0, "Parameter maxComponents must be > 0.")

    // Assemble input features into a single vector
    val featuresCol = "features"
    val pcaFeaturesCol = "pcaFeatures"
    var inputData: DataFrame = data
    if (assembleFeatures) inputData = assemble(data, outputCol = featuresCol)

    // Perform PCA
    val pca = new PCA()
      .setInputCol(featuresCol)
      .setOutputCol(pcaFeaturesCol)
      .setK(maxComponents)
    val fittedModel = pca.fit(inputData)

    // Get the components accounting for the given explained variance
    val variances =
      Utils.take(fittedModel.explainedVariance.toArray, explainedVariance)
    var numComponents = variances.length
    if (numComponents == 0) numComponents = maxComponents

    // Apply transformation
    val numRows = fittedModel.pc.numRows
    val numCols = fittedModel.pc.numCols
    val values = fittedModel.pc.values.clone().slice(0, numRows * numComponents)
    val pc = new DenseMatrix(numRows, numComponents, values)
    val transposed = pc.transpose
    val transformer = udf { vector: Vector => transposed.multiply(vector) }

    return inputData
      .withColumn(
        pcaFeaturesCol,
        transformer(col(featuresCol))
      )
      .drop(featuresCol)
      .withColumnRenamed(pcaFeaturesCol, featuresCol)
  }

  /** Bins the given column values according to the defined splits */
  def binning(
      data: DataFrame,
      columnName: String,
      splits: Array[Double]
  ): DataFrame = {
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
      inputCols: Option[Array[String]] = None,
      outputCol: String
  ): DataFrame = {
    var assembler = new VectorAssembler()
    inputCols match {
      case Some(value) => assembler = assembler.setInputCols(value)
      case None        => assembler = assembler.setInputCols(data.columns)
    }
    return assembler
      .setOutputCol(outputCol)
      .transform(data)
  }

  def standardize(
      data: DataFrame,
      inputCol: String
  ): DataFrame = {
    return new StandardScaler()
      .setWithMean(true)
      .setWithStd(true)
      .setInputCol(inputCol)
      .setOutputCol(s"T${inputCol}")
      .fit(data)
      .transform(data)
      .drop(inputCol)
      .withColumnRenamed(s"T${inputCol}", inputCol)
  }

}
