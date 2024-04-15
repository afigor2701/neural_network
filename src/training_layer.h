#pragma once

#include "include_eigen.h"

namespace NNeuralNetwork {

class TLayer;

class TTrainingLayer {
public:
    explicit TTrainingLayer(TLayer* layer_ptr, double learning_rate);

    MatrixXd Evaluate(MatrixXd x);

    MatrixXd PropagationAndCoeffUpdate(MatrixXd gradient);

private:
    TLayer* layer_ptr_;
    //MatrixXd gradient_for_A_;
    //VectorXd gradient_for_b_;
    MatrixXd last_in_;
    MatrixXd last_out_;
    double learning_rate_;
};

}  // namespace NNeuralNetwork
