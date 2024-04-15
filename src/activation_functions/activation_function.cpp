#include "activation_function.h"

namespace NNeuralNetwork {

TActivationFunction::TActivationFunction(FunctionPtr evaluate, FunctionPtr derivative) : evaluate_(evaluate), derivative_(derivative) {
}

double TActivationFunction::Evaluate(double x) const {
    return evaluate_(x);
}
double TActivationFunction::Derivative(double x) const {
    return derivative_(x);
}

VectorXd TActivationFunction::Evaluate(const VectorXd& x) const {
    VectorXd result(x.size());
    for (Index i = 0; i < x.size(); ++i) {
        result(i) = evaluate_(x(i));
    }
    return result;
}

MatrixXd TActivationFunction::DerivativeMatrix(const VectorXd& x) const {
    MatrixXd result = MatrixXd::Zero(x.size(), x.size());
    for (Index i = 0; i < x.size(); ++i) {
        result(i, i) = derivative_(x(i));
    }
    return result;
}

}  // namespace NNeuralNetwork
