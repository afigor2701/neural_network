#include "training_layer.h"

#include "layer.h"

#include <stdexcept>

namespace NNeuralNetwork {

TTrainingLayer::TTrainingLayer(TLayer* layer_ptr, double learning_rate) : layer_ptr_(layer_ptr), learning_rate_(learning_rate) {
}

MatrixXd TTrainingLayer::Evaluate(MatrixXd x) {
    last_in_ = x;
    return last_out_ = layer_ptr_->Evaluate(std::move(x));
}

MatrixXd TTrainingLayer::PropagationAndCoeffUpdate(MatrixXd gradient) {
    throw std::runtime_error("Not Implemented");
}

}  // NNeuralNetwork;
