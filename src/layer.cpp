#include "layer.h"

#include "utils/random.h"
#include <iostream>

namespace NNeuralNetwork {

TLayer::TLayer(Index in_size, Index out_size, TActivationFunction function)
    : A_(utils::RandomGenerator::GetRandomMatrix(out_size, in_size) / 20.0), b_(utils::RandomGenerator::GetRandomVector(out_size) / 20.0), function_(function) {
}

MatrixXd TLayer::Evaluate(MatrixXd x) const {
    return function_.Evaluate((A_ * x).colwise() + b_);
}

}  // namespace NNeuralNetwork
