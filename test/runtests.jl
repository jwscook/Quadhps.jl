using Quadhps
using SpecialFunctions, QuadGK, Test

@testset "Quadhps tests" begin
  @testset "Test vs QuadGK" begin
    f(x) = 1 ./ (1 .+ x.^3)
    lower = 0.1
    upper = 10.0
    a = Quadhps.quadhp(x->f(x), lower, upper)[1]
    b = QuadGK.quadgk(x->f(x), lower, upper)[1]
    @test a ≈ b atol=2 * eps()
  end

  @testset "Types" begin
    f(x) = 1 ./ (1 .+ x.^3)
    lower = BigFloat(0.1)
    upper = 10.0
    a = Quadhps.quadhp(x->f(x), lower, upper)[1]
    b = QuadGK.quadgk(x->f(x), lower, upper)[1]
    @test a ≈ b atol=2 * eps()
  end

  @testset "Testing accuracy" begin
    a = Quadhps.quadhp(x->1.0, 0.0, 2.0, rtol=2*eps())
    @test a[1] ≈ 2.0 atol=10 * eps()
    @test a[2] ≈ 0.0 atol=10 * eps()
    b = Quadhps.quadhp(x->x, 1.0, 2.0, rtol=2*eps())
    @test b[1] ≈ 1.5 atol=10 * eps()
    @test b[2] ≈ 0.0 atol=10 * eps()
    c = Quadhps.quadhp(x->3 .* x.^2 .+ 1.0, 2.0, 5.0, rtol=120*eps())
    @test c[1] ≈ 120.0 atol=1000 * eps()
    @test c[2] ≈ 0.0 atol=1000 * eps()
  end

  @testset "Test intervals" begin
    g(x::Float64) = exp(-x^2)*besselj(1, x)*besselj(-1, x) * x^4
    g(x::Vector{T}) where {T} = g.(x)
 
    Quadhps.quadhp(x->[1, 2]*x, 0.0, 1.0)
    a = Quadhps.quadhp(x->3 .* x.^2 .+ 1.0, 0.0, 5.0, rtol=120*eps())
    b = Quadhps.quadhp(x->3 .* x.^2 .+ 1.0, 0.0, 2.0, 5.0, rtol=120*eps())
    c = Quadhps.quadhp(x->3 .* x.^2 .+ 1.0, 0.0, 2.0, 4.0, 5.0, rtol=120*eps())
    @test a[1] ≈ b[1]
    @test a[1] ≈ c[1]
  end
end
