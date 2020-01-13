
const Data = [(0, 0)
							(0, 1)
							(1, 0)
							(1, 1)]

function and(x1::Int64, x2::Int64)
				x = [x1, x2]
				w = [0.5, 0.5]
				b = -0.7
				sum(x .* w) + b <= 0 ? 0 : 1
end

function nand(x1::Int64, x2::Int64)
				x = [x1, x2]
				w = [-0.5, -0.5]
				b = 0.7
				sum(x .* w) + b <= 0 ? 0 : 1
end

function or(x1::Int64, x2::Int64)
				x = [x1, x2]
				w = [0.5, 0.5]
				b = -0.2
				sum(x .* w) + b <= 0 ? 0 : 1
end

function xor(x1::Int64, x2::Int64)
				and(or(x1, x2), nand(x1, x2))
end

function testing(func)
				for i in Data
								print("$i:\t")
								println(func(i[1], i[2]))
				end
end

