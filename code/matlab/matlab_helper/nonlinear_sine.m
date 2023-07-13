function A = nonlinear_sine(X, w_0)
%NONLINEAR applies the nonlinear function to the input vector x. In this
%case the Sine-function is applied.
%   x: input vector (1x?)
%   w_0: weight or constant
%   A: returns sine(w_0 * x), which is the vector of the activations of
%   that layer

% sine of x times w_0 (ref. to SIREN)
A = sin(w_0 * X);
end