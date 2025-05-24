fun main() {
    val input = mutableListOf(1.0, 2.0, 3.0)
    var target = mutableListOf(0.0, 1.0, 0.5)
    val nn = NeuralNetwork(
        inputSize = 3,
        hiddenLayers = listOf(4, 5),
        outputSize = 3,
        activationFunction = ::sigmoid
    )
    nn.train(listOf(input), listOf(target), epochs = 5, learningRate = 0.01)
    println("Output: ${nn.predict(input)}")

}