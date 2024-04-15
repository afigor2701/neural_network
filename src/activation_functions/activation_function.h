#pragma once

#include "../include_eigen.h"

namespace NNeuralNetwork {

class TActivationFunction {
public:
    using Signature = double(double);
    using FunctionPtr = Signature*;

    template <typename TFunction>
    TActivationFunction(TFunction) : evaluate_(TFunction::Evaluate), derivative_(TFunction::Derivative) {
    }

    TActivationFunction(FunctionPtr evaluate, FunctionPtr derivative);

    double Evaluate(double x) const;
    double Derivative(double x) const;

    // Скорее всего надо принимать Eigen::MatrixBase, чтобы избежать копирований, т.к. буду делать слайся матрицы
    VectorXd Evaluate(const VectorXd& x) const;
    MatrixXd DerivativeMatrix(const VectorXd& x) const;

private:
    FunctionPtr evaluate_;
    FunctionPtr derivative_;
};

}  // namepsace NNeuralNetwork

/*class TActivationFunction {
public:
    template <typename TFunction>
    TActivationFunction(const TFunction& function) : function_ptr_(std::make_unique<TConcept<Function>>(function)) {
    }

    virtual double Evaluate(double x) const;
    virtual double Derivative(double x) const;

    // Скорее всего надо принимать Eigen::MatrixBase, чтобы избежать копирований, т.к. буду делать слайся матрицы
    virtual VectorXd Evaluate(const VectorXd& x) const;
    virtual MatrixXd DerivativeMatrix(const VectorXd& x) const;

private:
    class TBaseConcept {
    public:
        virtual double Evaluate(double x) const = 0;
        virtual double Derivative(double x) const = 0;

        virtual VectorXd Evaluate(const VectorXd& x) const = 0;
        virtual MatrixXd DerivativeMatrix(const VectorXd& x) const = 0;
    };

    template <typename TFunction>
    class TConcept : TBaseConcept {
    public:
        virtual double Evaluate(double x) const {
            return function_.Evaluate(x);
        }
        virtual double Derivative(double x) const {
            return funciton_.Derivative(x);
        }

        virtual VectorXd Evaluate(const VectorXd& x) const {
            return function_.Evaluate(x);
        }
        virtual MatrixXd DerivativeMatrix(const VectorXd& x) const {
            return function_.DerivativeMatrix(x);
        }

    private:
        TFunction function_;
    };

private:
    std::unique_ptr<TBaseConcept> function_;
};

}  // namespace NNeuralNetwork*/
