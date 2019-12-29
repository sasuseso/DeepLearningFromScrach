module DeepLearningFromScrach

step_function(x::Vector) = x .> 0

sigmoid(x::Array{Float64, 2}) = 1 ./ (1 .+ exp.(-x))

rectified_linear_unit(x) = max.(0, x)

identity_function(x) = x

softmax(a::Array) = exp.(a .- maximum(a)) / sum(exp.(a .- maximum(a)))

end # module
