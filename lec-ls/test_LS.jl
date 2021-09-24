using LinearAlgebra;
using Plots;
using LaTeXStrings;

include("Cheby_LS.jl");

# parameters that can be played with
d = 100;
# dimension
κ = 10^2; # condition number
maxiter = 500; # maximum number of iterations

σ = 1.;
L = κ;

# general problem instance
A = randn(d,d);
A = A' * A;
lambdas, U = eigen(A);
lambdas = LinRange(σ, L, d);
A = U * diagm(lambdas) * U';

b = randn(d);
w = A \ b; # groundtruth

w0 = randn(d); # fix initializer
algs = ["Richardson" "Cheby" "Polyak" "ConjGrad"];
W = zeros(d, length(algs)); # can compare to the groundtruth w
pl = plot(title="dim = $d, kappa = $κ")
for ii = 1:length(algs)
    W[:,ii], obj = Cheby_LS(A, b, σ, L, algs[ii], w0, maxiter);
    plot!(pl, 1:length(obj), obj, label=algs[ii], lw=2, xlabel="iteration", ylabel=L"\|A\mathbf{w} - \mathbf{b}\|_2")
end
pl
savefig(pl,"Cheby-$d-$κ.png")
