function [respCell, correctCell] = simulate_WM_pers(pars,stimvaluesCell,corrrespCell,condVec,subjrespCell)
%SIMULATE_WM simulates data given parameters and stimuli
% 
% ========================= INPUT VARIABLES ============================
% PARS: 1x2 vector of parameters [epsilon, lambda]
%       epsilon: lapse rate
%       lambda: WM decay rate
%       pers: perseveration
%       tau: perseveration decay rate
% STIMVALUESCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars 
%       corresponding to the index of the stimulus presented on each trial
% CORRRESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of lanth nTrials of scalars
%       corresponding to the index of the correct button response
% CONDVEC: not used
% SUBJRESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to the participant's response
%
% ========================= OUTPUT VARIABLES ============================
% RESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to the index of the simulated button response
% CORRECTCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to whether the simulated data was rewarded for
%       correct response. 
%
% written by aspen yoo, november 2020
% aspen.yoo@berkeley.edu

% pars
epsilon = pars(1);
lambda = pars(2);
pers = pars(3);
tau = pars(4);
beta_WM = pars(end);

nBlocks = length(stimvaluesCell);

[respCell, correctCell] = deal(cell(1,nBlocks));
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    
    % data
    corrrespVec = corrrespCell{iblock};
    
    Cval = ones(1,3)./1/3;
    [respVec,correctVec] = deal(nan(1,nTrials));
    valueMat = ones(nStim,3)./3; % initializing all values for all stim and responses
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        
        p = exp( beta_WM*( valueMat(istim,:) - pers.*Cval ) ); 
        p=p/sum(p);
        p = (1-epsilon).*p + epsilon*1/3;
        if find(stimVec == istim,1,'first') == itrial
            ichoice = subjrespCell{iblock}(itrial);
        else
            ichoice = find([0 cumsum(p)]<rand,1,'last');
        end
        r = (ichoice == corrrespVec(itrial));
        
        valueMat = (1-lambda).*1/3 + lambda.*valueMat; % decay
        if (ichoice ~= -1)
            % updating value
            valueMat(istim,ichoice) = r; % updating current choice value
            
            % updating choice trace (perseveration)
            delta_choice = 0-Cval;
            delta_choice(ichoice) = 1-Cval(ichoice);
            Cval = Cval + tau.*delta_choice;
        end
        
        % saving variables
        respVec(itrial) = ichoice;
        correctVec(itrial) = r;
    end
    
    respCell{iblock} = respVec;
    correctCell{iblock} = correctVec;
end