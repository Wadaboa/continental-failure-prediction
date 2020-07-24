package evaluation

import preprocessing.Preprocessor

import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.linalg.{Vector, SparseVector}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import breeze.linalg.{DenseVector => BDV}

object EuclideanInertia {

  /** Computes the Within Set Sum of Squared Errors */
  def computeInertiaScore(
      data: Dataset[_],
      featuresCol: String,
      clusterCenters: Array[Vector]
  ): Double = {
    val bClusterCenters = data.sparkSession.sparkContext.broadcast(
      clusterCenters.map(p => new VectorWithNorm(OldVectors.fromML(p)))
    )
    val cost = data
      .select(featuresCol)
      .rdd
      .map { row =>
        pointCost(
          bClusterCenters.value,
          new VectorWithNorm(OldVectors.fromML(row.getAs[Vector](0)))
        )
      }
      .sum()
    bClusterCenters.destroy()
    cost
  }

  /** Returns the cost of a given point against the given cluster centers */
  def pointCost(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm
  ): Double = {
    findClosest(centers, point)._2
  }

  /** Returns the index of the closest center to the given point, as well as the cost */
  def findClosest(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm
  ): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      val currentDistance = distance(center, point)
      if (currentDistance < bestDistance) {
        bestDistance = currentDistance
        bestIndex = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /** Returns the squared Euclidean distance between two vectors */
  def distance(
      v1: VectorWithNorm,
      v2: VectorWithNorm,
      precision: Double = 1e-6
  ): Double = {
    val n = v1.vector.size
    require(v2.vector.size == n)
    require(v1.norm >= 0.0 && v2.norm >= 0.0)
    val sumSquaredNorm = v1.norm * v1.norm + v2.norm * v2.norm
    val normDiff = v1.norm - v2.norm
    var sqDist = 0.0
    val precisionBound1 =
      2.0 * EPSILON * sumSquaredNorm / (normDiff * normDiff + EPSILON)
    val v1Array = new BDV(v1.toDense.vector.toArray)
    val v2Array = new BDV(v2.toDense.vector.toArray)
    if (precisionBound1 < precision) {
      sqDist = sumSquaredNorm - 2.0 * (v1Array dot v2Array)
    } else if (
      v1.vector.isInstanceOf[SparseVector] || v2.vector
        .isInstanceOf[SparseVector]
    ) {
      val dotValue = v1Array dot v2Array
      sqDist = math.max(sumSquaredNorm - 2.0 * dotValue, 0.0)
      val precisionBound2 =
        EPSILON * (sumSquaredNorm + 2.0 * math.abs(dotValue)) /
          (sqDist + EPSILON)
      if (precisionBound2 > precision) {
        sqDist = OldVectors.sqdist(v1.vector, v2.vector)
      }
    } else {
      sqDist = OldVectors.sqdist(v1.vector, v2.vector)
    }
    sqDist
  }

}

object EuclideanGap {

  /** Computes the gap statistic score */
  def computeGapScore(
      data: DataFrame,
      featuresCol: String,
      predictionCol: String,
      clusterCenters: Array[Vector],
      numRandom: Int = 30
  ): Tuple2[Double, Double] = {
    val inertia: Double = math.log(
      EuclideanInertia.computeInertiaScore(data, featuresCol, clusterCenters)
    )
    val (expectedInertia, standardDeviation) = computeExpectedInertia(
      data,
      featuresCol,
      predictionCol,
      numRandom,
      clusterCenters.length
    )
    return (
      expectedInertia - inertia,
      computeGapDeviation(standardDeviation, numRandom)
    )
  }

  /** Returns the expected inertia values, as well as the standard deviation
    * of the inertia values, calculated over the generated random data
    */
  def computeExpectedInertia(
      data: DataFrame,
      featuresCol: String,
      predictionCol: String,
      numRandom: Int,
      k: Int
  ): Tuple2[Double, Double] = {
    val randomModel: KMeans = new KMeans()
      .setK(k)
      .setDistanceMeasure("euclidean")
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predictionCol)
    var randomInertiaValues: Array[Double] = Array()
    for (i <- (1 to numRandom)) {
      var randomData =
        getRandomData(
          data.drop(featuresCol, predictionCol)
        )
      randomData = Preprocessor.assemble(randomData, outputCol = featuresCol)
      var trainedRandomModel: KMeansModel = randomModel.fit(randomData)
      var randomPredictions = trainedRandomModel.transform(randomData)
      var randomInertia = EuclideanInertia.computeInertiaScore(
        randomPredictions,
        featuresCol,
        trainedRandomModel.clusterCenters
      )
      randomInertiaValues = randomInertiaValues :+ math.log(randomInertia)
    }
    return (
      randomInertiaValues.sum / numRandom.toDouble,
      stdDev(randomInertiaValues)
    )
  }

  /** Computes the standard error of the standard deviation of the inertia values,
    * calculated over the generated random data
    */
  def computeGapDeviation(standardDeviation: Double, numRandom: Int): Double = {
    return standardDeviation * math.sqrt(1.0 + (1.0 / numRandom.toDouble))
  }

  /** Generates random data in a uniform distribution, based on
    * the initial dataset's minimum and maximum values in each column
    */
  def getRandomData(data: DataFrame): DataFrame = {
    val minValues = rowToArrayOfDouble(data.groupBy().min().head)
    val maxValues = rowToArrayOfDouble(data.groupBy().max().head)
    var randomData = data.select("*")
    (minValues, maxValues, data.columns).zipped.foreach { (a, b, c) =>
      randomData = randomData.withColumn(c, rand() * (b - a) + a)
    }
    return randomData
  }

}
