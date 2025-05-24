import kotlin.math.exp
import kotlin.random.Random

fun sigmoid(x: Double): Double {
    return 1.0 / (1.0 + exp(-x))
}

fun sumWeights(input: MutableList<Double>, weights: MutableList<Double>): Double {
    var sum = weights[0]
    for (i in 1..<input.size) {
        sum += input[i] * weights[i]
    }
    return sum
}


class Neuron(
    var input: MutableList<Double>,
    var activationFunction: (Double) -> Double = ::sigmoid,
) {
    var output: Double = 0.0
    var error: Double = 0.0
    var weights: MutableList<Double> = MutableList(input.size) { 0.0 }

    init {
        // Since it is a creation of a neuron, wights will have random values
        for (i in weights.indices) {
            weights[i] = Random.nextDouble(-1.0, 1.0)
        }
    }

    fun feedForward() {
        val weightedSum = sumWeights(input, weights)
        output = activationFunction(weightedSum)
    }

    fun calculateError(target: Double) {
        error = target - output
    }

    fun updateWeights(learningRate: Double) {
        for (i in weights.indices) {
            weights[i] += learningRate * error * input[i]
        }
    }
}