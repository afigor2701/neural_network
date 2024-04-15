#pragma once

#include "../eigen/Eigen/Eigen"

namespace NNeuralNetwork {

// TODO: выбрать, как лучше, чтобы вектора лежали в памяти. Видимо, придется идти по векторам
// в матрице, поэтому возможно лучше расположить вектора по колонками в памяти.
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Index;

}  // namespace NNeuralNetwork
