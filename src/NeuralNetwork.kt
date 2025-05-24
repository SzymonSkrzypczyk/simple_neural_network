class NeuralNetwork(
    private val inputSize: Int,
    private val hiddenLayers: List<Int>,
    private val outputSize: Int,
    private val activationFunction: (Double) -> Double = ::sigmoid
) {
    private val layers: MutableList<MutableList<Neuron>> = mutableListOf()

    init {
        var previousSize = inputSize
        for (layerSize in hiddenLayers) {
            val layer = MutableList(layerSize) {
                Neuron(MutableList(previousSize) { 0.0 }, activationFunction)
            }
            layers.add(layer)
            previousSize = layerSize
        }
        val outputLayer = MutableList(outputSize) {
            Neuron(MutableList(previousSize) { 0.0 }, activationFunction)
        }
        layers.add(outputLayer)
    }

    fun feedForward(input: MutableList<Double>): MutableList<Double> {
        var currentInput = input
        for (layer in layers) {
            for (neuron in layer) {
                neuron.input = currentInput
                neuron.feedForward()
            }
            currentInput = layer.map { it.output }.toMutableList()
        }
        return currentInput
    }

    fun backpropagate(target: MutableList<Double>, learningRate: Double) {
        val outputLayer = layers.last()

        for (i in outputLayer.indices) {
            outputLayer[i].calculateError(target[i])
        }

        for (i in layers.size - 2 downTo 0) {
            val currentLayer = layers[i]
            val nextLayer = layers[i + 1]

            for (j in currentLayer.indices) {
                val neuron = currentLayer[j]
                val sum = nextLayer.sumOf { nextNeuron ->
                    if (j + 1 < nextNeuron.weights.size) {
                        nextNeuron.error * nextNeuron.weights[j + 1]
                    } else 0.0
                }
                neuron.error = sum * neuron.output * (1 - neuron.output)
            }
        }

        for (layer in layers) {
            for (neuron in layer) {
                neuron.updateWeights(learningRate)
            }
        }
    }

    fun train(inputs: List<MutableList<Double>>, targets: List<MutableList<Double>>, epochs: Int, learningRate: Double) {
        for (epoch in 0 until epochs) {
            for (i in inputs.indices) {
                feedForward(inputs[i])
                backpropagate(targets[i], learningRate)
            }
        }
    }

    fun predict(input: MutableList<Double>): MutableList<Double> {
        return feedForward(input)
    }
}