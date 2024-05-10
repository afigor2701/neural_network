add_library(Neural_Network
            layer.cpp
            activation_functions/activation_function.cpp
            activation_functions/ReLu.cpp
            activation_functions/sigmoid.cpp
            loss_functions/loss_function.cpp
            loss_functions/square_error.cpp
            utils/random.cpp
            neural_network.cpp
            neural_network_builder.cpp
            training_layer.cpp)

target_include_directories(Neural_Network PRIVATE ../third_party/eigen)

