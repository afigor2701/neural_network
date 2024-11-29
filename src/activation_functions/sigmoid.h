#pragma once

namespace NNeuralNetwork {

struct Sigmoid {
    static double Evaluate(double x);

    static double Derivative(double x);
};

}  // namespace NNeuralNetwork
