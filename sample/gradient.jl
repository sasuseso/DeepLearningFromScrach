using DeepLearningFromScrach
using Random

f1(x) = 0.01.*x.^2 .+ 0.1.*x
function numerical_diff(f, x)
								h = 1e-4
								(f(x+h) - f(x-h)) / 2h
end

function tangent_line(f, x)
								d = numerical_diff(f, x)
								println(d)
								y = f(x) - d*x

								t -> d*t + y
end

f2(x) = x[1]^2 + x[2]^2
function numerical_gradient(f, x::Array{Float64})
								h = 1e-4
								grad = zeros(size(x))

								for i=1:length(x)
																tmp = x[i]

																x[i] = tmp + h
																fxh1 = f(x)

																x[i] = tmp - h
																fxh2 = f(x)

																grad[i] = (fxh1 - fxh2) / 2h
																x[i] = tmp
								end
								grad
end

function gradient_descent(f::Function; init_x::Vector{Float64}, lr=0.01, step_num=100)
								x = init_x

								for i=1:step_num
																grad = numerical_gradient(f, x)
																x -= lr * grad
								end
								x
end

struct SimpleNet
								W::Array{Float64, 2}
								function SimpleNet()
																rng = MersenneTwister(1234)
																new(randn(rng, 2, 3))
								end
end

predict(SN::SimpleNet, x::Array{Float64}) = x * SN.W 

function loss(SN::SimpleNet, x::Array{Float64, 2}, t::Vector{Int})
								z = predict(SN, x)
								y = softmax(z)

								cross_entropy_error(y, t)
end
