#!/bin/julia
using DeepLearningFromScrach, Distributions, MLDatasets
struct TwoLayerNet
								params::Dict{String, Union{Array{Float64, 2}, Vector{Float64}}}
								grads::Union{Nothing,
																					Dict{String, Union{Array{Float64, 2}, Array{Float64, 1}}}} 

								function TwoLayerNet(input_size::Int, hidden_size::Int, output_size::Int, weight_init_std=0.01)
																w1 = rand(Uniform(0, .1), input_size, hidden_size) .* weight_init_std
																b1 = zeros(hidden_size)

																w2 = rand(Uniform(0, .1), hidden_size, output_size) .* weight_init_std
																b2 = zeros(output_size)

																new(Dict("w1" => w1, "w2" => w2, "b1" => b1, "b2" => b2), nothing)
								end
end

function predict(tln::TwoLayerNet, X::AbstractArray)
								x = reshape(X, 784, 60000)
								w1, w2 = tln.params["w1"]', tln.params["w2"]'
								b1, b2 = tln.params["b1"], tln.params["b2"]

								a1 = w1 * x .+ b1
								z1 = sigmoid(a1)
								a2 = (w2 * z1 .+ b2)'

								softmax(a2)
end

loss(tln::TwoLayerNet, x::Vector{Float64}, t::Array{Bool, 2}) = cross_entropy_error(predict(tln, x), t)

function accuracy(tln::TwoLayerNet,
																		x::AbstractArray,
																		t::Array{Bool, 2})
								y = argmax(predict(tln, x), dims=2)
								t = argmax(t, dims=2)

								sum(y .== t) / size(x)[1]
end

function onehot(t::Vector{Int})
								t_tr2 = zeros(Bool, 60000, 10)
								for i=1:length(t)
																t_tr2[i, t[i]+1] = true
								end
								t_tr2
end

function numerical_gradient(tln::TwoLayerNet, x, t)
								loss_w = w -> loss(tln, x, t)
								tln.grads = Dict(
																									"w1" => numerical_gradient(tln, loss_w, tln.params["w1"]),
																									"b1" => numerical_gradient(tln, loss_w, tln.params["b1"]),
																									"w2" => numerical_gradient(tln, loss_w, tln.params["w1"]),
																									"b2" => numerical_gradient(tln, loss_w, tln.params["b1"])
																								)
								tln.grads
end

function main()
								tln = TwoLayerNet(784, 100, 10)
end

main()
