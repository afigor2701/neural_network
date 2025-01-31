#include "sigmoid.h"

#include <cmath>

namespace NNeuralNetwork {

double Sigmoid::Evaluate(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Sigmoid::Derivative(double x) {
    double tmp = exp(-x);
    return 1.0 / (1.0 / tmp + 2.0 + tmp);
    //return tmp / ((1 + tmp) * (1 + tmp));
}

}  // namespace NNeuralNetwork
