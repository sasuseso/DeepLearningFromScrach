using MLDatasets, ImageView, PyCall, DeepLearningFromScrach
@pyimport pickle

function get_data()
				xtrain, ttrain = MNIST.traindata()
				xtest, ttest = MNIST.testdata()
				return xtest, ttest
end

function init_network()
				return pickle.loads(pybytes(read("/home/sasuseso/Scripts/sample_weight.pkl")))
end

function predict(network, X)
				W1, W2, W3 = network["W1"]', network["W2"]', network["W3"]'
				b1, b2, b3 = network["b1"], network["b2"], network["b3"]
				x = vec(X)

				a1 = W1 * x + b1
				z1 = sigmoid(a1)

				a2 = W2 * z1 + b2
				z2 = sigmoid(a2)

				a3 = W3 * z2 + b3

				y = softmax(a3)

				y
end

# cross_entropy_error(y, t) = -sum(t' .* log.(y')) / size(y)[1]
cross_entropy_error(y::Array{Float64}, t::Vector{Int}) = -sum(t' .* log.(y'[collect(1:size(y)[1]), t])) / size(y)[1]
cross_entropy_error(y::Array{Float64, 2}, t::Array{Bool, 2}) = -sum(t' .* log.(y')) / size(y)[1]
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i = 1:10000
								y = predict(network, x[:, :, i])
								p = argmax(vec(y))
								if p-1 == t[i]
																global accuracy_cnt += 1
								end
end

println("Accuracy: $(accuracy_cnt / 10000)")
