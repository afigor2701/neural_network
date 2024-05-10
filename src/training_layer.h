#pragma once

#include "include_eigen.h"
#include "layer.h"

namespace NNeuralNetwork {

class TTrainingLayer : public TLayer {
public:
    TTrainingLayer(TLayer layer_ptr, double learning_rate);

    MatrixXd Evaluate(MatrixXd x);

    MatrixXd PropagationAndCoeffUpdate(MatrixXd gradients);

private:
    MatrixXd last_in_;
    MatrixXd last_in_after_linear_;
    double learning_rate_;
};

}  // namespace NNeuralNetwork
