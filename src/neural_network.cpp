#include "neural_network.h"

#include "layer.h"
#include "training_layer.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include <iostream>

namespace NNeuralNetwork {

namespace {

std::vector<TTrainingLayer> MakeTrainingLayers(std::vector<TLayer>& layers, double learning_rate) {
    std::vector<TTrainingLayer> training_layers;
    training_layers.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        training_layers.emplace_back(std::move(layers[i]), learning_rate);
    }
    return training_layers;
}

std::vector<TLayer> MakeLayers(std::vector<TTrainingLayer>& training_layers) {
    std::vector<TLayer> layers;
    layers.reserve(training_layers.size());
    for (size_t i = 0; i < training_layers.size(); ++i) {
        layers.emplace_back(std::move(training_layers[i]));
    }
    return layers;
}

inline auto GetBatch(const MatrixXd& x, Index start, Index batch_size) {
    return x(all, seq(start, std::min(x.cols() - 1, start + batch_size - 1)));
}

}  // namespace

void TNeuralNetwork::Train(const MatrixXd& x, const MatrixXd& y, Index batch_size, TLossFunction loss_function,
                           double target_loss, Index max_epoch, double learning_rate) {
    assert(x.cols() == y.cols());
    assert(batch_size > 0);

    layers_ = MakeTrainingLayers(GetLayers(), learning_rate); // TODO: make guard

    std::cout << "Training started" << std::endl;

    for (size_t epoch = 0; epoch < max_epoch; ++epoch) {
        double loss = Epoch(x, y, batch_size, loss_function);
        std::cout << "Epoch: " << epoch + 1 << ", loss: " << loss << std::endl;
        if (loss <= target_loss) {
            break;
        }
    }

    layers_ = MakeLayers(GetTrainingLayers());
}

MatrixXd TNeuralNetwork::Predict(MatrixXd x) const {
    for (auto& layer : GetLayers()) {
        x = layer.Evaluate(std::move(x));
    }
    return x;
}

TNeuralNetwork::TNeuralNetwork(std::vector<TLayer> layers) : layers_(std::move(layers)) {
}

std::vector<TLayer>& TNeuralNetwork::GetLayers() {
    assert(std::holds_alternative<std::vector<TLayer>>(layers_));
    return std::get<std::vector<TLayer>>(layers_);
}

const std::vector<TLayer>& TNeuralNetwork::GetLayers() const {
    assert(std::holds_alternative<std::vector<TLayer>>(layers_));
    return std::get<std::vector<TLayer>>(layers_);
}

std::vector<TTrainingLayer>& TNeuralNetwork::GetTrainingLayers() {
    assert(std::holds_alternative<std::vector<TTrainingLayer>>(layers_));
    return std::get<std::vector<TTrainingLayer>>(layers_);
}

MatrixXd TNeuralNetwork::TrainingPredict(MatrixXd x) {
    for (auto& layer : GetTrainingLayers()) {
        x = layer.Evaluate(std::move(x));
    }
    return x;
}

double TNeuralNetwork::Epoch(const MatrixXd& x, const MatrixXd& y, Index batch_size, TLossFunction loss_function) {
    assert(std::holds_alternative<std::vector<TTrainingLayer>>(layers_));

    double loss = 0;
    for (Index i = 0; i < x.cols(); i += batch_size) {
        auto predicted =
            TrainingPredict(GetBatch(x, i, batch_size));

        auto curr_y_batch = GetBatch(y, i, batch_size);
        loss += loss_function.Evaluate(predicted, curr_y_batch);
        auto gradient = loss_function.DerivativeWithRespectForX(predicted, curr_y_batch);
        Propagation(std::move(gradient));
    }
    return loss;
}

void TNeuralNetwork::Propagation(MatrixXd gradient) {
    for (auto layer_it = GetTrainingLayers().rbegin(); layer_it != GetTrainingLayers().rend(); ++layer_it) {
        gradient = layer_it->PropagationAndCoeffUpdate(std::move(gradient));
    }
}


}  // namespace NNeuralNetwork
