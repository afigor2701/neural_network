#include "neural_network.h"

#include "layer.h"
#include "training_layer.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace NNeuralNetwork {

namespace {

std::vector<TTrainingLayer> MakeTrainingLayers(std::vector<TLayer>& layers, double learning_rate) {
    std::vector<TTrainingLayer> training_layers;
    training_layers.reserve(layers.size());
    for (size_t i = 0; i < training_layers.size(); ++i) {
        training_layers.emplace_back(&layers[i], learning_rate);
    }
    return training_layers;
}

MatrixXd TrainigPredict(MatrixXd x, std::vector<TTrainingLayer>& training_layers) {
    for (auto& training_layer : training_layers) {
        x = training_layer.Evaluate(std::move(x));
    }
    return x;
}

void Propagation(MatrixXd gradient, std::vector<TTrainingLayer>& training_layers) {
    for (auto layer_it = training_layers.rbegin(); layer_it != training_layers.rend(); ++layer_it) {
        gradient = layer_it->PropagationAndCoeffUpdate(std::move(gradient));
    }
}

}  // namespace


// TODO
void TNeuralNetwork::Train(const MatrixXd& x, const MatrixXd& y, Index batch_size, TLossFunction loss_function,
                           double target_loss, Index max_epoch, double learning_rate) {
    assert(x.rows() == y.rows() && x.cols() == y.cols());
    assert(batch_size > 0);

    std::vector<TTrainingLayer> training_layers = MakeTrainingLayers(layers_, learning_rate);

    for (size_t epoch = 0; epoch < max_epoch; ++epoch) {
        //double loss = Epoch();
    }


    throw std::runtime_error("Not Implemented");
}

MatrixXd TNeuralNetwork::Predict(MatrixXd x) const {
    for (auto& layer : layers_) {
        x = layer.Evaluate(std::move(x));
    }
    return x;
}

// TODO
void TNeuralNetwork::ResetWeights() {
    throw std::runtime_error("Not Implemented");
}


TNeuralNetwork::TNeuralNetwork(std::vector<TLayer> layers) : layers_(std::move(layers)) {
}

double TNeuralNetwork::Epoch(const MatrixXd& x, const MatrixXd& y, std::vector<TTrainingLayer>& training_layers,
                             Index batch_size, TLossFunction loss_function) {
    double loss = 0;
    for (Index i = 0; i < x.cols(); i += batch_size) {
        auto predicted = 
            TrainigPredict(x(Eigen::indexing::all, Eigen::indexing::seq(i, std::min(x.cols() - 1, i + batch_size - 1))),
                           training_layers);
        
        loss += loss_function.Evaluate(predicted, y);

        auto gradient = loss_function.DerivativeWithRespectForX(predicted, y);
        Propagation(gradient, training_layers);
    }
    return loss;
}


}  // namespace NNeuralNetwork
