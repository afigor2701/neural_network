#pragma once

namespace NNeuralNetwork {

struct SquareError {

    static double Evaluate(double x, double y);

    static double DerivativeWithRespectForX(double x, double y);

};

}  // namespace NNeuralNetwork
