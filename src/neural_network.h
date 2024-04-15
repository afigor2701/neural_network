#pragma once


#include "include_eigen.h"
#include "loss_functions/loss_function.h"

#include <vector>

namespace NNeuralNetwork {

class TNeuralNetworkBuilder;
class TLayer;
class TTrainingLayer;

class TNeuralNetwork {
public:
    void Train(const MatrixXd& x, const MatrixXd& y, Index batch_size, TLossFunction loss_function, double target_loss,
               Index max_epoch, double learning_rate);

    MatrixXd Predict(MatrixXd x) const;

    void ResetWeights();

private:
    // TNeuralNetworkBuilder
    TNeuralNetwork(std::vector<TLayer> layers);
    friend class TNeuralNetworkBuilder;

    double Epoch(const MatrixXd& x, const MatrixXd& y, std::vector<TTrainingLayer>& training_layers, Index batch_size,
                 TLossFunction loss_function);

private:
    std::vector<TLayer> layers_;
};

}  // namespace NNeuralNetwork
