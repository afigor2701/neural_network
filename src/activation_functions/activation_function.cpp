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

MatrixXd TActivationFunction::Evaluate(MatrixXd x) const {
    for (Index i = 0; i < x.rows(); ++i) {
        for (Index j = 0; j < x.cols(); ++j) {
            x(i, j) = evaluate_(x(i, j));
        }
    }
    return x;
}

VectorXd TActivationFunction::DerivativeMatrix(const VectorXd& x) const {
    VectorXd result = VectorXd::Zero(x.size());
    for (Index i = 0; i < x.size(); ++i) {
        result(i) = derivative_(x(i));
    }
    return result;
}

}  // namespace NNeuralNetwork
