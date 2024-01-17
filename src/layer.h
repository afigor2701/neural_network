#pragma once

#include "include_eigen.h"

namespace NNeuralNetwork {

class TLayer {
public:
    TLayer(size_t size, TActivationFunction function);

    MatrixXd Evaluate(const MatrixXd& x) const;
    VectorXd Derivative(const VectorXd& x, const VectorXd& u) const;

    void AccumulateChangeWeights(const VectorXd& x, const VectorXd& u);
    void ApplyChanges();

private:
    VectorXd A_;
    VectorXd b_;
    TActivationFunction function_;
    VectorXd gradient_for_A_;
    VectorXd gradient_for_b_;
};


}  // namespace NNeuralNetwork
