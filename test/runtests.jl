using Quadhps
using ForwardDiff, QuadGK, Test

@testset "Quadhps tests" begin

  @testset "Basic" begin
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

  @testset "Versus QuadGK" begin
    f(x) = 1 ./ (1 .- x.^2)
    lower = -0.999
    upper = 0.999
    a = Quadhps.quadhp(x->f(x), lower, upper, rtol=2*eps())[1]
    b = QuadGK.quadgk(x->f(x), lower, upper, rtol=2*eps())[1]
    @test a ≈ b atol=1000 * eps()
  end

  @testset "Types" begin
    f(x) = x
    lower = BigFloat(0.1)
    upper = 10.0
    a = Quadhps.quadhp(x->f(x), lower, upper)[1]
    b = QuadGK.quadgk(x->f(x), lower, upper)[1]
    @test a ≈ b rtol=eps()
  end

  @testset "Vectorised" begin
    a = Quadhps.quadhp(x->[1, 2], 0.0, 1.0)[1]
    @test a[1] ≈ 1 atol=2 * eps()
    @test a[2] ≈ 2 atol=2 * eps()
  end

  @testset "Intervals" begin
    a = Quadhps.quadhp(x->3 .* x.^2 .+ 1.0, 0.0, 5.0, rtol=120*eps())
    b = Quadhps.quadhp(x->3 .* x.^2 .+ 1.0, 0.0, 2.0, 5.0, rtol=120*eps())
    c = Quadhps.quadhp(x->3 .* x.^2 .+ 1.0, 0.0, 2.0, 4.0, 5.0, rtol=120*eps())
    @test a[1] ≈ b[1]
    @test a[1] ≈ c[1]
  end

  @testset "ForwadDiff" begin
    integrand = sin
    differentiand(x) = Quadhps.quadhp(y->integrand(y[1]), 0.0, x[1])[1]
    b = π/2
    @test integrand(b) ≈ ForwardDiff.gradient(differentiand, [b])[1]
    a = ForwardDiff.gradient(x->
      Quadhps.quadhp(y->sin(y[1]*x[1]), 0.0, 1.0)[1],
          [0.5])[1]
    @test a ≈ 0.469181324769897
  end
end
