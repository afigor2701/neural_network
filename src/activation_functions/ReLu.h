#pragma once

namespace NNeuralNetwork {

struct ReLu {
    static double Evaluate(double x);

    static double Derivative(double x);
};

}  // namespace NNeuralNetwork
