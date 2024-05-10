#include "random.h"

namespace NNeuralNetwork::utils {

MatrixXd RandomGenerator::GetRandomMatrix(Index a, Index b) {
    return GetGenerator().norm_gen.template generate<MatrixXd>(a, b, GetGenerator().urng);
}

VectorXd RandomGenerator::GetRandomVector(Index a) {
    return GetGenerator().norm_gen.template generate<MatrixXd>(a, 1, GetGenerator().urng);
}

RandomGenerator& RandomGenerator::GetGenerator() {
    static RandomGenerator random_generator;
    return random_generator;
}

}
