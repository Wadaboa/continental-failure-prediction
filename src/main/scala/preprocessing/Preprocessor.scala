package preprocessing

import org.apache.spark.ml.feature.{
  QuantileDiscretizer,
  Bucketizer,
  PCA,
  PCAModel,
  VectorAssembler
}
import org.apache.spark.ml.linalg.{Vector, DenseMatrix}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Column
import org.apache.spark.sql.functions.{trim, when, length, col, udf}

object Preprocessor {

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
      maxComponents: Integer,
      assemble: Boolean = true,
      explainedVariance: Double = 0.35
  ): DataFrame = {
    require(
      explainedVariance > 0 && explainedVariance <= 1,
      "Invalid value for the explainedVariance parameter."
    )

    // Assemble input features into a single vector
    var inputData = data.select("*")
    val featuresCol = "features"
    val pcaFeaturesCol = "pcaFeatures"
    if (assemble) {
      inputData = new VectorAssembler()
        .setInputCols(data.columns)
        .setOutputCol(featuresCol)
        .transform(inputData)
    }

    // Perform PCA
    val pca = new PCA()
      .setInputCol(featuresCol)
      .setOutputCol(pcaFeaturesCol)
      .setK(maxComponents)
    val fittedModel = pca.fit(inputData)

    // Get the components accounting for the given explained variance
    val variances =
      take(fittedModel.explainedVariance.toArray, explainedVariance)
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
      .select(pcaFeaturesCol)
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

}
