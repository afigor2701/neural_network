#include "square_error.h"

namespace NNeuralNetwork {

double SquareError::Evaluate(double x, double y) {
    return (x - y) * (x - y);
}

double SquareError::DerivativeWithRespectForX(double x, double y) {
    return 2.0L * (x - y);
}

}  // namespace NNeuralNetwork
