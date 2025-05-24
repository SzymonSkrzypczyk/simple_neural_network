import kotlin.math.pow


object Utils {
    fun MeanSquaredError(target: MutableList<Double>, predicted: MutableList<Double>): Double {
        var sum = 0.0
        for (i in target.indices) {
            sum += (target[i] - predicted[i]).pow(2)
        }
        return sum / target.size
    }
}