module DeepLearningFromScrach
export step_function,
    sigmoid, rectified_linear_unit, identity_function, softmax, cross_entropy_error
include("gradient.jl")
include("layers.jl")

step_function(x::Vector) = x .> 0

sigmoid(x) = 1 ./ (1 .+ exp.(-x))

rectified_linear_unit(x) = max.(0, x)

identity_function(x) = x

function softmax(x::AbstractArray{N,2}) where {N}
    x = x'
    x .-= maximum(x, dims = 1)

    (exp.(x) ./ sum(exp.(x), dims = 1))'
end

softmax(a::AbstractVector) = exp.(a .- maximum(a)) / sum(exp.(a .- maximum(a)))

function cross_entropy_error(y::AbstractArray, t::AbstractVector)
    batch_size = size(y)[1]
    Y = zip.(collect(1:batch_size), t)
    result = [y[Tuple(i)[1][1], Tuple(i)[1][2] + 1] .+ 1e-7 for i in Y]
    -(sum(log.(result) .* t) / batch_size)
end

cross_entropy_error(y::AbstractArray, t::AbstractArray) =
    cross_entropy_error(y, argmax(t, dims = 2))

cross_entropy_error(y::Vector{Float64}, t::Int) = sum((y - t)^2) .* 0.5
end # module


