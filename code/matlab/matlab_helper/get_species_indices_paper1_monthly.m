function ind=swiftai_get_species_indices_paper1_monthly
% initialize structure ind, containing the hard coded indices of the SWIFT
% AI Neural Network. The order the NN expects the inputs.
%
% Author:   H. Mohn, January 2023
ind.Cly=1;
ind.Bry=2;
ind.NOy=3;
ind.H2O=4;
ind.O3=5;

ind.z=6;

ind.overhead=7;
ind.t=8;
ind.daylight=9;

ind.O2_reaction_coef=10;
ind.O3_reaction_coef=11;
ind.ClOy_reaction_coef=12;
ind.ClOx_reaction_coef=13;

% Total inputs N_x
ind.N_x=13;