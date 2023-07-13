function dOx = predict_paper1_monthly(x, ptnet, mean_dox, std_dox)
%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Neural Network %
%%%%%%%%%%%%%%%%%%%%%%%%
%PREDICT by using Multilayer Perceptron ( 5-layer)
%   x: input vector (1x11)
%   ptnet: contains all matrices of the NN-layers
%   month_idx: e.g. 1 for january
%   y_pred: returns the non-normalied prediction for x
w0_initial = 6;
w0 = 4;

%disp('layer 1')
tmp = linear_vectorized(x, ptnet.layer_model_0_weight, ptnet.layer_model_0_bias);
tmp = nonlinear_sine( tmp, w0_initial );

%disp('layer 2')
tmp = linear_vectorized(tmp, ptnet.layer_model_1_weight, ptnet.layer_model_1_bias);
tmp = nonlinear_sine( tmp, w0 );

%disp('layer 3')
tmp = linear_vectorized(tmp, ptnet.layer_model_2_weight, ptnet.layer_model_2_bias);
tmp = nonlinear_sine( tmp, w0 );

%disp('layer 4')
tmp = linear_vectorized(tmp, ptnet.layer_model_3_weight, ptnet.layer_model_3_bias);
tmp = nonlinear_sine( tmp, w0 );

%disp('layer 5')
tmp = linear_vectorized(tmp, ptnet.layer_model_4_weight, ptnet.layer_model_4_bias);
tmp = nonlinear_sine( tmp, w0 );

%disp('layer 6')
tmp = linear_vectorized(tmp, ptnet.layer_model_5_weight, ptnet.layer_model_5_bias);

y_pred = tmp;

%%%%%%%%%%%%%%%%%%%%%%
% Undo Normalization %
%%%%%%%%%%%%%%%%%%%%%%
% (old matlab): copy all rows to have the same number as the data-set
mean_dox = repmat( mean_dox, 1, length(y_pred) );
dOx = y_pred * std_dox + mean_dox;
end