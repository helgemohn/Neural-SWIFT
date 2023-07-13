clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural-SWIFT - Apply testdata to Neural-SWIFT      %
% Helge Mohn 2023                                    %            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
month_idx = 5;
num_testing_data = 1e5;

filename_testing_data = sprintf('../../data/testing/SWIFT-AI_test_month_%02d.nc', month_idx);
fprintf('Neural-SWIFT                           : Dataset %s\n', filename_testing_data)

path_matlab_models = '../../models/matlab/';
model_name = sprintf( 'month_%02d', month_idx );

addpath('matlab_helper');
addpath(path_matlab_models);
neuralswift_type = 'paper1_monthly';

     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load matrices of the neural network   %             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load matrices of the neural network
fprintf('Neural-SWIFT                           : Model %s%s_%02d.mat\n', path_matlab_models, neuralswift_type, month_idx);
      
for month_idx = 1:12
    neuralswift_model_dox.(sprintf('month_%02d', month_idx)) = load(sprintf('%s%s_%02d.mat', path_matlab_models, neuralswift_type, month_idx));
end

fprintf('Neural-SWIFT                           : %s matrices loaded\n', neuralswift_type)
         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Testing Data                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load testing data
data = ncread(filename_testing_data, 'data');

% indices of the neural network;
ind_nn=get_species_indices_paper1_monthly();

X=NaN(ind_nn.N_x,num_testing_data); 

% current_date
current_date = data(1,1:num_testing_data);

% The chemical families
X(ind_nn.Cly,:)=data(21,1:num_testing_data)';
X(ind_nn.Bry,:)=data(22,1:num_testing_data)';
X(ind_nn.NOy,:)=data(23,1:num_testing_data)';
X(ind_nn.H2O,:)=data(24,1:num_testing_data)'; % Use H2O as HOy
X(ind_nn.O3,:)=data(25,1:num_testing_data)';  % Use O3 as Ox

% Log pressure height
X(ind_nn.z,:)=data(17,1:num_testing_data)';

% Overhead ozone
X(ind_nn.overhead,:)=data(5,1:num_testing_data)';

% Temperature
X(ind_nn.t,:)=data(18,1:num_testing_data)';

% daylight
X(ind_nn.daylight,:)=data(8,1:num_testing_data)';

% Reaction Coef
X(ind_nn.O2_reaction_coef,:)=data(9,1:num_testing_data)';
X(ind_nn.O3_reaction_coef,:)=data(10,1:num_testing_data)';
X(ind_nn.ClOy_reaction_coef,:)=data(13,1:num_testing_data)';
X(ind_nn.ClOx_reaction_coef,:)=data(14,1:num_testing_data)';
fprintf('Neural-SWIFT                           : Testing data loaded (%i samples)\n', num_testing_data)

%%%%%%%%%%%%%%%%%%%%
% Normalize Data   %
%%%%%%%%%%%%%%%%%%%%
[X, mean_dox, std_dox] = normalize_paper1_monthly( X );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Neural Network & undo normalization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
dox_out = predict_paper1_monthly( X, neuralswift_model_dox.(model_name), mean_dox, std_dox );
toc

dOx_fullchem = data(26,1:num_testing_data);
fprintf('MSE: %0.2e \n', mean((dox_out - dOx_fullchem).^2))
fprintf('RMSE: %0.2e \n', mean((dox_out - dOx_fullchem).^2)^0.5)
fprintf('MAE: %0.2e \n', mean(abs(dox_out - dOx_fullchem)))

%%%%%%%%%%%%%%%%%%%%
% Update Ox        %
%%%%%%%%%%%%%%%%%%%%
ox_new = X(ind_nn.O3,:) + dox_out;

