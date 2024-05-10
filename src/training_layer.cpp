#include "training_layer.h"

#include "layer.h"

#include <stdexcept>
#include <iostream>

namespace NNeuralNetwork {

TTrainingLayer::TTrainingLayer(TLayer layer, double learning_rate) : TLayer(std::move(layer)), learning_rate_(learning_rate) {
}

MatrixXd TTrainingLayer::Evaluate(MatrixXd x) {
    last_in_ = x;
    last_in_after_linear_ = (A_ * x).colwise() + b_;
    return function_.Evaluate(last_in_after_linear_);
}

MatrixXd TTrainingLayer::PropagationAndCoeffUpdate(MatrixXd gradients) {
    MatrixXd gradient_for_A = MatrixXd::Zero(A_.rows(), A_.cols());
    VectorXd gradient_for_b = VectorXd::Zero(b_.size());

    MatrixXd new_gradients = MatrixXd::Zero(gradients.rows(), A_.cols());

    for (Index i = 0; i < gradients.rows(); ++i) {
        VectorXd dSigma = function_.DerivativeMatrix(last_in_after_linear_.col(i));
        auto tmp = gradients.row(i).transpose().cwiseProduct(dSigma);

        gradient_for_b += tmp;
        gradient_for_A += tmp * last_in_.col(i).transpose();
        new_gradients.row(i) = tmp.transpose() * A_;
    }

    gradient_for_A *= learning_rate_;
    gradient_for_b *= learning_rate_;
    A_ -= gradient_for_A;
    b_ -= gradient_for_b;

    return new_gradients;
}

}  // NNeuralNetwork;
