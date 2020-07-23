import org.apache.spark.mllib.linalg.{
  Vector => OldVector,
  Vectors => OldVectors
}

package object evaluation {

  /** Returns machine precision value */
  lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0) {
      eps /= 2.0
    }
    eps
  }

  /** A vector with its norm for fast distance computation */
  class VectorWithNorm(val vector: OldVector, val norm: Double)
      extends Serializable {

    def this(vector: OldVector) = this(vector, OldVectors.norm(vector, 2.0))

    def this(array: Array[Double]) = this(OldVectors.dense(array))

    /** Converts the vector to a dense vector */
    def toDense: VectorWithNorm =
      new VectorWithNorm(OldVectors.dense(vector.toArray), norm)
  }

}
