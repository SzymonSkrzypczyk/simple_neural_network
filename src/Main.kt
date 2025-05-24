import kotlin.math.cos

fun main() {
    val input = generateSequence(0.0) { it + 0.05 }
        .takeWhile { it <= 12 }
        .toMutableList()
    val target = input.map { 0.5 * cos(0.2 * it * it) + 0.5}.toMutableList()
    val nn = NeuralNetwork(
        inputSize = input.size,
        hiddenLayers = listOf(4),
        outputSize = input.size,
        activationFunction = ::sigmoid
    )
    nn.train(listOf(input), listOf(target), epochs = 5, learningRate = 0.01)
    val predicted = nn.predict(input)
    println(Utils.MeanSquaredError(target, predicted))

}