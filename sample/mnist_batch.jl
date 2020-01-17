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

function predict(network, x)
				W1, W2, W3 = network["W1"]', network["W2"]', network["W3"]'
				b1, b2, b3 = network["b1"], network["b2"], network["b3"]

				a1 = W1 * x .+ b1
				z1 = sigmoid(a1)

				a2 = W2 * z1 .+ b2
				z2 = sigmoid(a2)

				a3 = W3 * z2 .+ b3

				y = softmax(a3)

				y
end

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i = 1:100:10000
								x_batch = reshape(x[:, :, i:i+batch_size-1], (784, 100))
								y_batch = predict(network, x_batch)
								p = [argmax(y_batch[:, i])-1 for i in 1:100]
								global accuracy_cnt += count(p .== t[i:i+batch_size-1])
end

println("Accuracy: $(accuracy_cnt / 10000)")
