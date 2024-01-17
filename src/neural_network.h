#pragma once

#include "../include_eigen.h"
#include "activation_function/activation_function.h"
#include "layer.h"

#include <initializer_list>
#include <tuple>
#include <vector>

namespace NNeuralNetwork {

template <typename TLossFunciton>
class TNeuralNetwork {
public:
    TNeuralNetwork(const std::initializer_list<std::pair<size_t, TActivationFunction>>& layers,
                   TLossFunction loss_funciton);

    void Train(const MatrixXd& x, const MatrixXd& results);

    MatrixXd Predict(const MatrixXd& x) const;

    void ResetWeights();

private:
    std::vector<TLayer> layers_;
    TLossFunction loss_funciton_;
};

template <typename TLossFunciton>
TNeuralNetwork<TLossFunciton>::TNeuralNetwork(const std::initializer_list<std::pair<size_t, TActivationFunction>>& layers,
                                              TLossFunction loss_funciton)
                                              : loss_funciton_(std::move(loss_funciton)) {
    layers_.reserve(layers.size());
    for (size_t layer : layers) {
        layers_.emplace_back(layer.first, layer.second);
    }
}

}  // namespace NNeuralNetwork
