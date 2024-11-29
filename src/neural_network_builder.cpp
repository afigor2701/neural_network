#include "neural_network_builder.h"

#include "layer.h"

namespace NNeuralNetwork {

TNeuralNetworkBuilder::TNeuralNetworkBuilder(Index in_size) {
    layers_sizes_.push_back(in_size);
}

TNeuralNetworkBuilder& TNeuralNetworkBuilder::AddLayer(Index size, TActivationFunction function) {
    layers_sizes_.push_back(size);
    activation_functions.push_back(function);
    return *this;
}

TNeuralNetwork TNeuralNetworkBuilder::Finish() {
    std::vector<TLayer> layers;

    for (Index i = 0; i < activation_functions.size(); ++i) {
        layers.emplace_back(layers_sizes_[i], layers_sizes_[i + 1], activation_functions[i]);
    }

    return TNeuralNetwork(std::move(layers));
}

}  // namespace NNeuralNetwork
