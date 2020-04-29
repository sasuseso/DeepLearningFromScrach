export Layer, ReLU

abstract type Layer end

mutable struct ReLU <: Layer
    mask::Union{BitVector, Nothing}

    ReLU() = new(nothing)
end

function forward(l::ReLU, x::AbstractArray)
    l.mask = (x .<= 0)
    out = copy(x)
    map(x -> out[x] = 0, findall(l.mask));
    out
end

function backward(l::ReLU, dout)
    map(x -> dout[x] = 0, findall(l.mask));
    dout
end
