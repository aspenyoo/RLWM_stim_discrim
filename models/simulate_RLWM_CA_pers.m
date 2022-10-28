function [respCell, correctCell] = simulate_RLWM_CA_pers(pars,stimvaluesCell,corrrespCell,condVec,subjrespCell)
%SIMULATE_RLWM_CA simulates data given parameters and stimuli
% 
% ========================= INPUT VARIABLES ============================
% PARS: 1x9 vector of parameters 
%    [alpha, neg_alpha, CA_e, CA_t ,epsilon, lambda, ns3, ns6, beta]
%       alpha: RL learning rate for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       CA_e, CA_t: credit assignment confusion parameter (for exemplar and text
%       conditions only)
%       epsilon: lapse rate
%       lambda: WM decay rate
%       ns3: WM (vs RL) weight for set size 3 blocks
%       ns6: WM weight for set size 6 blocks
%       beta: softmax inverse temperature parameter
% STIMVALUESCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars 
%       corresponding to the index of the stimulus presented on each trial
% CORRRESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of lanth nTrials of scalars
%       corresponding to the index of the correct button response
% CONDVEC: 1 x nBlocks vector, condition indices 
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
% alpha = pars(1);
% neg_alpha = pars(2);
% CAVec = pars(3:4);
% epsilon = pars(5);
% lambda = pars(6);
% ns = pars(7:8);
% beta_RL = pars(9);
% beta_WM = 100;
alpha = pars(1);
neg_alpha = pars(2);
CAVec = pars(3:4);
epsilon = pars(5);
lambda = pars(6);
ns = pars(7:8);
pers = pars(9);
tau = pars(10);
beta_RL = pars(11);
beta_WM = 100;

nBlocks = length(stimvaluesCell);

[respCell, correctCell] = deal(cell(1,nBlocks));
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    icond = condVec(iblock);
    
    switch icond
        case 1 % exemplar
            CA = CAVec(1);
        case 2 % category (no effect)
            CA = 0;
        case 3 % text
            CA = CAVec(2);
    end
    
    % data
    corrrespVec = corrrespCell{iblock};
    
    Cval = ones(1,3)./1/3;
    [respVec,correctVec] = deal(nan(1,nTrials));
    [QvalMat,WMvalMat] = deal(ones(nStim,3)./3); % initializing all values for all stim and responses    
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        
        % p of each choice
        p_RL = exp(beta_RL*(QvalMat(istim,:)-pers.*Cval)); p_RL=p_RL/sum(p_RL);
        p_WM = exp(beta_WM*WMvalMat(istim,:)-pers.*Cval); p_WM=p_WM/sum(p_WM);
        p = ns(nStim/3)*p_WM + (1-ns(nStim/3))*p_RL;
        p = (1-epsilon).*p + epsilon*1/3;
      
        if find(stimVec == istim,1,'first') == itrial;
            ichoice = subjrespCell{iblock}(itrial);
        else
            ichoice = find([0 cumsum(p)]<rand,1,'last');
        end
        r = (ichoice == corrrespVec(itrial));
        
        if (ichoice ~= -1)
            % updating
            stimvec = 1:nStim;
            stimvec(istim) = [];
            delta = r-QvalMat(istim,ichoice);
            if (r==1);
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + alpha*delta;
                QvalMat(stimvec,ichoice) = QvalMat(stimvec,ichoice) + alpha*CA*delta; % update other states because you're confused which state you're in
            else
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + neg_alpha*delta;
                QvalMat(stimvec,ichoice) = QvalMat(stimvec,ichoice) + neg_alpha*CA*delta;
            end
            WMvalMat = (1-lambda).*1/3 + lambda.*WMvalMat; % decay
            WMvalMat(istim,ichoice) = r;
            
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