function Cheby_LS(A, b, σ, L, alg="Cheby", w=randn(size(b)), maxiter=10^3, tol=1e-5)
# The Chebyshev algorithm for solving linear system Aw = b
    # A: square matrix, positive definite
    # b: vector
    # σ: lower bound of the spectrum of A
    # L: upper bound of the spectrum of A
    # w: initial guess
    # alg: "Richardson", "Cheby" (default), "Polyak", "ConjGrad"
    # maxiter: maximum number of iteration; default 10^3
    # tol: tolerance; default 1e-5

    z = copy(w); # backup the previous iterate
    g = A*z - b; # gradient

    ϵ = eps(0.0); # smallest number to prevent dividing by 0
    κ = L / (σ + ϵ); # condition number
    c = ( (κ-1) / (κ+1) / 2 )^2;
    η = 2. / (L+σ+ϵ); # step size
    γ = 2.; # momentum size
    if alg == "Richardson"
        c = 0;
    elseif alg == "Polyak"
        γ = 2*(κ+1) / (sqrt(κ)+1)^2;
    elseif alg == "ConjGrad"
        γ = 1.; # don't forget this!
        η = dot(g,g) / (dot(A*g,g)+ϵ);
        ζ = η; # backup step size
    end

    w = z - η*g; # Richardson step

    obj = zeros(maxiter); # let's roll
    for t = 1:maxiter
        obj[t] = norm(g); # evaluated at z, not w

        # mission accomplished
        if obj[t] <= tol
            return z, obj[1:t]
        end

        g = A*w - b;

        if alg == "ConjGrad"
            η = dot(g,g) / (dot(A*g,g)+ϵ);
            γ = 1. / ( 1 - η*dot(g,g) / (ζ*obj[t]^2*γ+ϵ) );
            ζ = η;
        else
            γ = 1. / (1 - c*γ + ϵ);
        end

        # mission halted
        if t == maxiter
            return z, obj
        end
        
        # momentum update and backup w
        w, z = w - γ*η*g + (γ-1)*(w - z), w;
    end

end
