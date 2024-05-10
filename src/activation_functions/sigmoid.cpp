#include "sigmoid.h"

#include <cmath>

namespace NNeuralNetwork {

double Sigmoid::Evaluate(double x) {
    return 1.0L / (1.0L + exp(-x));
}

double Sigmoid::Derivative(double x) {
    double tmp = exp(-x);
    return tmp / ((1 + tmp) * (1 + tmp));
}

}  // namespace NNeuralNetwork
