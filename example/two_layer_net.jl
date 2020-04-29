using DeepLearningFromScrach, Distributions, MLDatasets

struct TwoLayerNet
    params::Dict
    grads::Dict

    function TwoLayerNet(
        input_size::Int,
        hidden_size::Int,
        output_size::Int,
        weight_init_std = 0.01,
    )
        new(
            Dict(
                "W1" => weight_init_std .* randn(input_size, hidden_size),
                "b1" => zeros(hidden_size),
                "W2" => weight_init_std .* randn(hidden_size, output_size),
                "b2" => zeros(output_size),
            ),
            Dict(),
        )
    end
end

function predict(tln::TwoLayerNet, x)
    a1 = x * tln.params["W1"] .+ tln.params["b1"]
    z1 = sigmoid(a1)
    a2 = z1 * tln.params["W2"] .+ tln.params["b2"]'
    softmax(a2)
end

function loss(tln::TwoLayerNet, x, t)
    y = predict(tln, x)
    cross_entropy_error(y, t)
end

function numerical_gradient(tln, x, t)
    lossw(W) = loss(tln, x, t)
    Dict(
        "W1" => _numerical_gradient(lossw, tln.params["W1"]),
        "b1" => _numerical_gradient(lossw, tln.params["b1"]),
        "W2" => _numerical_gradient(lossw, tln.params["W2"]),
        "b2" => _numerical_gradient(lossw, tln.params["b2"]),
    )
end
