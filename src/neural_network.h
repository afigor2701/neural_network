#pragma once

#include "include_eigen.h"
#include "loss_functions/loss_function.h"

#include "training_layer.h"

#include <variant>
#include <vector>

namespace NNeuralNetwork {

class TNeuralNetworkBuilder;

class TNeuralNetwork {
public:
    void Train(const MatrixXd& x, const MatrixXd& y, Index batch_size, TLossFunction loss_function, double target_loss,
               Index max_epoch, double learning_rate);

    MatrixXd Predict(MatrixXd x) const;

private:
    // TNeuralNetworkBuilder
    TNeuralNetwork(std::vector<TLayer> layers);
    friend class TNeuralNetworkBuilder;

    std::vector<TLayer>& GetLayers();
    const std::vector<TLayer>& GetLayers() const;
    std::vector<TTrainingLayer>& GetTrainingLayers();

    MatrixXd TrainingPredict(MatrixXd x);

    double Epoch(const MatrixXd& x, const MatrixXd& y, Index batch_size, TLossFunction loss_function);
    
    void Propagation(MatrixXd gradient);

private:
    std::variant<std::vector<TLayer>, std::vector<TTrainingLayer>> layers_;
};

}  // namespace NNeuralNetwork
