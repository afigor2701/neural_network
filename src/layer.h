#pragma once

#include "activation_functions/activation_function.h"
#include "include_eigen.h"

namespace NNeuralNetwork {

class TLayer {
public:
    TLayer(Index in_size, Index out_size, TActivationFunction function);

    MatrixXd Evaluate(MatrixXd x) const;

private:
    MatrixXd A_;
    VectorXd b_;
    TActivationFunction function_;
};


}  // namespace NNeuralNetwork
