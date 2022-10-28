function [respCell, correctCell] = simulate_WM(pars,stimvaluesCell,corrrespCell,condVec,subjrespCell)
%SIMULATE_WM simulates data given parameters and stimuli
% 
% ========================= INPUT VARIABLES ============================
% PARS: 1x2 vector of parameters [epsilon, lambda]
%       epsilon: lapse rate
%       lambda: WM decay rate
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
beta = 100;%pars(3);

nBlocks = length(stimvaluesCell);

[respCell, correctCell] = deal(cell(1,nBlocks));
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    
    % data
    corrrespVec = corrrespCell{iblock};
    
    [respVec,correctVec] = deal(nan(1,nTrials));
    valueMat = ones(nStim,3)./3; % initializing all values for all stim and responses
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        
        p=exp(beta*valueMat(istim,:)); p=p/sum(p);
        p = (1-epsilon).*p + epsilon*1/3;
        if find(stimVec == istim,1,'first') == itrial;
            ichoice = subjrespCell{iblock}(itrial);
        else
            ichoice = find([0 cumsum(p)]<rand,1,'last');
        end
        r = (ichoice == corrrespVec(itrial));
        
        if (ichoice ~= -1)
            % updating value
            valueMat = (1-lambda).*1/3 + lambda.*valueMat; % decay
            valueMat(istim,ichoice) = r; % updating current choice value
        end
        
        % saving variables
        respVec(itrial) = ichoice;
        correctVec(itrial) = r;
    end
    
    respCell{iblock} = respVec;
    correctCell{iblock} = correctVec;
end