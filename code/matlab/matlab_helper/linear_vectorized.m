function Z = linear_vectorized(X, W, B)
%LINEAR applies a linear function to the input vector x.
%   x: input vector (1x?)
%   erg: returns x * weights + bias

B = repmat( B', 1, size(X, 2) );

Z = W * X + B;
end

