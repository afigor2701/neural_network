#pragma once

#include "../include_eigen.h"

namespace NNeuralNetwork {

class TLossFunction {
public:
    using Signature = double(double, double);
    using FunctionPtr = Signature*;

    template <typename TFunction>
    TLossFunction(TFunction) : evaluate_(TFunction::Evaluate), derivative_x_(TFunction::DerivativeWithRespectForX) {
    }

    TLossFunction(FunctionPtr evaluate, FunctionPtr derivative_x);

    double Evaluate(double x, double y) const;
    double DerivativeWithRespectForX(double x, double y) const;

    // Скорее всего надо принимать Eigen::MatrixBase, чтобы избежать копирований, т.к. буду делать слайся матрицы
    double Evaluate(const MatrixXd& x, const MatrixXd& y) const;
    MatrixXd DerivativeWithRespectForX(const MatrixXd& x, const MatrixXd& y) const;

private:
    FunctionPtr evaluate_;
    FunctionPtr derivative_x_;
};

}  // namepsace NNeuralNetwork