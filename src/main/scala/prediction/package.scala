import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.SparseVector
import breeze.linalg.{DenseVector => BDV}

package object prediction {

  def getRandomSeed(): Int = 42

  /**
    * The following code was taken from an old version of Spark
    * and adapted to work with Spark 3.0.0
    */

  /** Returns machine precision value */
  lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  /** A vector with its norm for fast distance computation */
  class VectorWithNorm(val vector: Vector, val norm: Double)
      extends Serializable {

    def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

    def this(array: Array[Double]) = this(Vectors.dense(array))

    /** Converts the vector to a dense vector */
    def toDense: VectorWithNorm =
      new VectorWithNorm(Vectors.dense(vector.toArray), norm)
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
        sqDist = Vectors.sqdist(v1.vector, v2.vector)
      }
    } else {
      sqDist = Vectors.sqdist(v1.vector, v2.vector)
    }
    sqDist
  }

}
