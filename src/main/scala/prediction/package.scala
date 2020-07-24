import scala.reflect.ClassTag

package object prediction {

  def getRandomSeed(): Int = 42

  def tuple2ToArray[T: ClassTag](t: Tuple2[T, T]): Array[T] = Array(t._1, t._2)

}
