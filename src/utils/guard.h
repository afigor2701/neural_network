

namespace NNeuralNetwork::utils {

template <Callback>
class Guard {
public:
    Guard(Callback&& callback) : callback_(std::move(callback)) {
    }

    ~Guard() {
        std::move(callback_)();
    }

private:
    Callback callback_;
};

}
