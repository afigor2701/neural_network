#include "../include_eigen.h"
#include "third_party/EigenRand/EigenRand/EigenRand"

namespace NNeuralNetwork::utils {

struct RandomGenerator {
public:

    static MatrixXd GetRandomMatrix(Index a, Index b);
    static VectorXd GetRandomVector(Index a);

private:
    RandomGenerator() = default;
    static RandomGenerator& GetGenerator();

private:
    Eigen::Rand::P8_mt19937_64 urng{ 54 };
    Eigen::Rand::NormalGen<double> norm_gen{ 0.0, 1.0 };

};

}
