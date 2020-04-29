mutable struct MulLayer
    x::Union{Nothing, Number}
    y::Union{Nothing, Number}

    MulLayer() = new(nothing, nothing)
end

function forward(mull::MulLayer, x::Number, y::Number) 
    mull.x = x
    mull.y = y
    x * y
end

function backward(mull::MulLayer, dout)
    dx = dout * mull.y
    dy = dout * mull.x
    dx, dy
end

struct AddLayer end
forward(addl::AddLayer, x, y) = x + y
backward(addl::AddLayer, dout) = dout, dout

function buy_apple()
    apple = 100
    apple_num = 2
    tax = 1.1
    
    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    apple_price = forward(mul_apple_layer, apple, apple_num)
    price = forward(mul_tax_layer, apple_price, tax)

    println(price)

    dprice = 1
    dapple_price, dtax = backward(mul_tax_layer, dprice)
    dapple, dapple_num = backward(mul_apple_layer, dapple_price)
    println("$dapple, $dapple_num, $dtax")
end

function buy_apple_orange()
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    apple_price = forward(mul_apple_layer, apple, apple_num)
    orange_price = forward(mul_orange_layer, orange, orange_num)
    all_price = forward(add_apple_orange_layer, apple_price, orange_price)
    price = forward(mul_tax_layer, all_price, tax)
    println("price: $price")

    dprice = 1
    dall_price, dtax = backward(mul_tax_layer, dprice)
    dapple_price, dorange_price = backward(add_apple_orange_layer, dall_price)
    dorange, dorange_num = backward(mul_orange_layer, dorange_price)
    dapple, dapple_num = backward(mul_apple_layer, dapple_price)
    println("$dapple_num, $dapple, $dorange_num, $dorange, $dtax")
end

if abspath(PROGRAM_FILE) == @__FILE__
    buy_apple_orange()
end
