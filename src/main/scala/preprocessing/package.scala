package object preprocessing {

  def take(l: Array[Double], limit: Double): Array[Double] = {
    var sum: Double = 0
    l.takeWhile { e =>
      sum += e
      sum <= limit
    }
  }

}
