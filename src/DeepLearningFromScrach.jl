module DeepLearningFromScrach
export step_function, sigmoid, rectified_linear_unit, identity_function, softmax

step_function(x::Vector) = x .> 0

sigmoid(x) = 1 ./ (1 .+ exp.(-x))

rectified_linear_unit(x) = max.(0, x)

identity_function(x) = x

softmax(a::Array) = exp.(a .- maximum(a)) / sum(exp.(a .- maximum(a)))

end # module
