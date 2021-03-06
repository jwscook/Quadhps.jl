module Quadhps
using FastGaussQuadrature, FastClosures, StaticArrays


struct _Helper{T}
  abstol::T
  reltol::T
  max_depth::Int
  split::Int
  function _Helper(abstol::T, reltol::T, max_depth::Int, split::Int
      ) where {T<:Real}
    @assert split > 0 && max_depth >= 0 "$split, $max_depth"
    return new{T}(abstol, reltol, max_depth, split)
  end
end

#const memo = Dict{Int64, Tuple{Array{Float64,1},Array{Float64,1},
#                               Array{Float64,1}}}()
const memo = Dict()

@inline function memoisedgausslengendre(n::Int)
  if !haskey(memo, n)
    x, w = FastGaussQuadrature.gausslegendre(n)
    memo[n] = (x, w)
  end
  return memo[n]
end

@inline function calculate_node_weight(h::_Helper, n::Int,
    a, b)
  x, w = memoisedgausslengendre(n)
  halfdifference = @. (b - a) / 2
  halfsum = @. (b + a) / 2
  y = @. x * halfdifference + halfsum
  return y, w, halfdifference
end

@inline function simdmapreduce(f::T, op, iter) where {T<:Function}
  @inbounds output = f(iter[1])
  @inbounds @simd for i ∈ 2:length(iter)
    output += f(iter[i])
  end
  return output
end

@inline function applyquadrature(f::T, op, x, w) where {T<:Function}
  @inbounds output = f(x[1]) * w[1]
  @inbounds @simd for i ∈ 2:length(w)
    output += f(x[i]) * w[i]
  end
  return output
end

@inline function quad(f::T, h::_Helper, a, b, n::Int
    ) where {T<:Function}
  x, w, normalisation = calculate_node_weight(h, n, a, b)
  return applyquadrature(f, +, x, w) * normalisation
end

@inline function quad(f::T, h::_Helper, a, b, c, n::Int
    ) where {T<:Function}
  @assert a < b < c "$a, $b, $c"
  U = promote_type(typeof(a), typeof(b), typeof(c))
  @assert isapprox(b .- a, c .- b, rtol=sqrt(eps(U)), atol=0) "$a, $b, $c"
  x, w, normalisation = calculate_node_weight(h, n, a, b)
  b_a = b .- a
  reductionop = @closure x -> (f(x) .+ f(x .+ b_a))
  return applyquadrature(reductionop, +, x, w) * normalisation
end

@inline function hasconverged(a, b, h::_Helper, l::Int,
    nrm::T) where {T<:Function}
  all(@. nrm(a - b) < h.split^l * h.reltol * nrm(a + b) / 2) && return true
  all(@. nrm(a + b) / 2 <= h.abstol) && return true
  l > h.max_depth && return true
  return false
end

function quadhp(f::T, h::_Helper, a, b,
    n::Int, l::Int, nrm::U) where {T<:Function, U<:Function}
  @assert -Inf < a < b < Inf "a = $a, b = $b"
  @assert ispow2(n) "n is $n"
  qp = quad(f, h, a, b, n)
  qh = quad(f, h, a, (b .+ a) / 2, b, Int(n / 2))
  hasconverged(qp, qh, h, l, nrm) && return @SVector [(qp + qh) / 2, @. nrm(qp .- qh)]
  reductionop = @closure i -> quadhp(f, h, a .+ i * (b .- a) / h.split,
    a .+ (i + 1) * (b .- a) / h.split, 2n, l + 1, nrm)
  return simdmapreduce(reductionop, +, 0:(h.split - 1))
end

const default_order = 32
const default_split = 2
const default_max_depth = 8

function quadhp(f::T, a, b; norm::U=abs,
                rtol::Real=eps(), atol::Real=0.0,
                order::Int=default_order, max_depth::Int=default_max_depth,
                split::Int=default_split) where {T<:Function, U<:Function}
  return quadhp(f, _Helper(atol, rtol, max_depth, split), a, b, order, 0, norm)
end

function quadhp(f::T, ab...; norm::U=abs,
                rtol::Real=eps(), atol::Real=0.0,
                order::Int=default_order, max_depth::Int=default_max_depth,
                split::Int=default_split) where {T<:Function, U<:Function}
  reductionop = @closure i -> quadhp(f, ab[i], ab[i+1], norm=norm, rtol=rtol,
    atol=atol, order=order, max_depth=max_depth, split=split)
  return simdmapreduce(reductionop, +, 1:(length(ab)-1))
end

end # module
