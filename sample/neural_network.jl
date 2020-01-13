function init_network()
				return Dict(
										"W1" => [0.1 0.3 0.5; 0.2 0.4 0.6],
										"b1" => [0.1 0.2 0.3],
										"W2" => [0.1 0.4; 0.2 0.5; 0.3 0.6],
										"b2" => [0.1 0.2],
										"W3" => [0.1 0.3; 0.2 0.4],
										"b3" => [0.1 0.2]
										)
end

function forward(network, x)
				W1, W2, W3 = network["W1"], network["W2"], network["W3"]
				b1, b2, b3 = network["b1"], network["b2"], network["b3"]

				a1 = x * W1 + b1
				z1 = sigmoid(a1)
				a2 = z1 * W2 + b2
				z2 = sigmoid(a2)
				a3 = z2 * W3 + b3
				return identity_function(a3)
end


forward(init_network(), [1.0 0.5])
