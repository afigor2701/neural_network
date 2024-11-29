#include "ReLu.h"

namespace NNeuralNetwork {

double ReLu::Evaluate(double x) {
    return (x >= 0.0 ? x : 0.0);
}

double ReLu::Derivative(double x) {
    return (x >= 0.0 ? 1.0 : 0.0);
}

}  // namespace NNeuralNetwork
