#include "activation_function.h"

namespace NNeuralNetwork {

virtual double TActivationFunction::Evaluate(double x) const {
    return function_->Evaluate(x);
}
virtual double TActivationFunction::Derivative(double x) const {
    return funciton_->Derivative(x);
}

virtual VectorXd TActivationFunction::Evaluate(const VectorXd& x) const {
    return funciton_->Evaluate(x);
}
virtual MatrixXd TActivationFunction::DerivativeMatrix(const VectorXd& x) const {
    return function_->DerivativeMatrix(x);
}

}  // namespace NNeuralNetwork
