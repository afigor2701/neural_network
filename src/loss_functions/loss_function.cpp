#include "loss_function.h"

#include <cassert>

namespace NNeuralNetwork {

TLossFunction::TLossFunction(FunctionPtr evaluate, FunctionPtr derivative_x) : evaluate_(evaluate), derivative_x_(derivative_x) {
}

double TLossFunction::Evaluate(double x, double y) const {
    return evaluate_(x, y);
}
double TLossFunction::DerivativeWithRespectForX(double x, double y) const {
    return derivative_x_(x, y);
}

double TLossFunction::Evaluate(const MatrixXd& x, const MatrixXd& y) const {
    assert(x.rows() == y.rows() && x.cols() == y.cols());
    double loss = 0;
    for (Index i = 0; i < x.cols(); ++i) {
        double loss_for_vector = 0;
        for (Index j = 0; j < x.rows(); ++j) {
            loss_for_vector += evaluate_(x(j, i), y(j, i));
        }
        loss += loss_for_vector / x.rows();
    }
    return loss;
}

MatrixXd TLossFunction::DerivativeWithRespectForX(const MatrixXd& x, const MatrixXd& y) const {
    assert(x.rows() == y.rows() && x.cols() == y.cols());

    MatrixXd result = MatrixXd::Zero(x.cols(), x.rows());
    for (Index i = 0; i < x.cols(); ++i) {
        for (Index j = 0; j < x.rows(); ++j) {
            result(i, j) = derivative_x_(x(j, i), y(j, i));
        }
    }
    return result;
}

}  // namespace NNeuralNetwork
