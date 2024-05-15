#include "include_eigen.h"
#include "neural_network_builder.h"
#include "neural_network.h"

#include "third_party/mnist/include/mnist/mnist_reader.hpp"

#include "activation_functions/sigmoid.h"
#include "activation_functions/ReLu.h"

#include "loss_functions/square_error.h"

using namespace NNeuralNetwork;

namespace {

struct DataSet {
    MatrixXd training_data;
    MatrixXd training_ans;
    MatrixXd test_data;
    MatrixXd test_ans;
    Index in_size;
    Index out_size;
};

}  // namespace

MatrixXd ConvertImagesToMatrixXd(const auto& images) {
    MatrixXd result = MatrixXd::Zero(images[0].size(), images.size());
    for (Index i = 0; i < images.size(); ++i) {
        const auto& picture = images[i];

        for (Index pixl = 0; pixl < picture.size(); ++pixl) {
            result(pixl, i) = picture[pixl] / 255.0;
        }
    }
    return result;
}

MatrixXd ConvertLabesToMatrixXd(const auto& labels) {
    MatrixXd result = MatrixXd::Zero(10, labels.size());
    for (Index i = 0; i < result.cols(); ++i) {
        result(static_cast<Index>(labels[i]), i) = 1.0;
    }
    return result;
}

DataSet GetMnistData() {
    DataSet dataset;
    auto mnist_data = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../third_party/mnist");

    dataset.training_data = ConvertImagesToMatrixXd(mnist_data.training_images);
    dataset.training_ans = ConvertLabesToMatrixXd(mnist_data.training_labels);

    dataset.test_data = ConvertImagesToMatrixXd(mnist_data.test_images);
    dataset.test_ans = ConvertLabesToMatrixXd(mnist_data.test_labels);

    dataset.in_size = dataset.training_data.rows();
    dataset.out_size = 10;

    return dataset;
}

size_t GetAccuracy(const TNeuralNetwork& net, const MatrixXd& x, const MatrixXd& y) {
    size_t result = 0;

    auto predicted = net.Predict(x);
    for (Index i = 0; i < predicted.cols(); ++i) {
        Index predict;
        predicted.col(i).maxCoeff(&predict);
        Index correct;
        y.col(i).maxCoeff(&correct);
        result += (predict == correct);
    }
    return result;
}

int main() {
    auto dataset = GetMnistData();

    TNeuralNetwork net = TNeuralNetworkBuilder{dataset.in_size}.AddLayer(100, ReLu{})
                                                               .AddLayer(100, ReLu{})
                                                               .AddLayer(dataset.out_size, Sigmoid{})
                                                               .Finish();
    net.Train(dataset.training_data, dataset.training_ans, 75, SquareError{}, 0, 100, 0.005);

    size_t a = GetAccuracy(net, dataset.test_data, dataset.test_ans);
    std::cout << "Correct: " << a << " " << "out of " << dataset.test_data.cols() << ". Accuracy rate: " << static_cast<double>(a) / dataset.test_data.cols() << std::endl;
}
