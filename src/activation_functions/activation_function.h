#pragma once

#include "../include_eigen.h"

namespace NNeuralNetwork {

class TActivationFunction {
public:
    using Signature = double(double);
    using FunctionPtr = Signature*;

    template <typename TFunction>
    TActivationFunction(TFunction) : evaluate_(TFunction::Evaluate), derivative_(TFunction::Derivative) {
    }

    TActivationFunction(FunctionPtr evaluate, FunctionPtr derivative);

    double Evaluate(double x) const;
    double Derivative(double x) const;

    MatrixXd Evaluate(MatrixXd x) const;
    VectorXd DerivativeMatrix(const VectorXd& x) const;

private:
    FunctionPtr evaluate_;
    FunctionPtr derivative_;
};

}  // namepsace NNeuralNetwork
