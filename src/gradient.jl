export _numerical_gradient

function _numerical_gradient(f, x::AbstractArray)
    h = 1e-4
    grad = zeros(size(x))

    for i in CartesianIndices(x)
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

