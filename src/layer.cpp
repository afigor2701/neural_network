#include "layer.h"

namespace NNeuralNetwork {

// Изменить инициализацию на рандомные из N(0, 1)
TLayer::TLayer(Index in_size, Index out_size, TActivationFunction function)
    : A_(MatrixXd::Zero(out_size, in_size)), b_(VectorXd::Zero(out_size)), function_(function) {
}

MatrixXd TLayer::Evaluate(MatrixXd x) const {
    x = A_ * x;
    for (Index i = 0; i < x.cols(); ++i) {
        x.col(i) += b_;
        x.col(i) = function_.Evaluate(x.col(i));
    }
    return x;
}

}  // namespace NNeuralNetwork
