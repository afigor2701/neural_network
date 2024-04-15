#pragma once

/*
    Example of usage:
        TNeuralNetwork net = TNeuralNetworkBuilder{3}.AddLayer(5, Sigmoid{}).AddLayer(4, ReLu{}).Finish();
*/

#include "activation_functions/activation_function.h"
#include "include_eigen.h"
#include "neural_network.h"

#include <vector>

namespace NNeuralNetwork {

class TNeuralNetworkBuilder {
public:
    TNeuralNetworkBuilder(Index in_size);

    TNeuralNetworkBuilder& AddLayer(Index size, TActivationFunction function);

    TNeuralNetwork Finish() &&;

private:
    std::vector<Index> layers_sizes_;
    std::vector<TActivationFunction> activation_functions;
};

}  // namespace NNeuralNetwork