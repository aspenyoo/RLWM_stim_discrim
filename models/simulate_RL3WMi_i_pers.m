function [respCell, correctCell, QvalCell] = simulate_RL3WMi_i_pers(pars,stimvaluesCell,corrrespCell,condVec,subjrespCell)
%SIMULATE_RL3WM simulates data given parameters and stimuli
% 
% ========================= INPUT VARIABLES ============================
% PARS: 1x11 vector of parameters 
%    [alpha_e, alpha_c, alpha_t, neg_alpha, epsilon, lambda, ns3, ns6, pers, tau, beta]
%       alpha_e: RL learning rate for exemplar condition for positive feedback
%       alpha_c: RL learning rate for category condition for positive feedback
%       alpha_t: RL learning rate for text condition for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       epsilon: lapse rate
%       lambda: WM decay rate
%       ns3: WM (vs RL) weight for set size 3 blocks
%       ns6: WM weight for set size 6 blocks
%       pers: perseveration
%       tau: perseveration decay rate
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
% QVALCELL: 1xnBlocks cell. each containing nSxnResponses matrix of Q
%       values after learning block
%
% written by aspen yoo, november 2020
% aspen.yoo@berkeley.edu

% pars
alpha = pars(1:3);
neg_alpha = pars(4);
epsilon = pars(5);
lambda = pars(6);
ns = pars(7:8);
pers = pars(9);
tau = pars(10);
imult = pars(11:13);
beta_RL = pars(end);
beta_WM = 100;

nBlocks = length(stimvaluesCell);

[respCell, correctCell, QvalCell] = deal(cell(1,nBlocks));
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    icond = condVec(iblock);
    
    % data
    corrrespVec = corrrespCell{iblock};
    
    Cval = ones(1,3)./1/3;
    [respVec,correctVec] = deal(nan(1,nTrials));
    [QvalMat,WMvalMat] = deal(ones(nStim,3)./3); % initializing all values for all stim and responses    
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        
        % p of each choice
        p_RL = exp( beta_RL*( QvalMat(istim,:) - pers.*Cval ) ); 
        p_RL=p_RL/sum(p_RL);
        p_WM = exp( beta_WM*( WMvalMat(istim,:) - pers.*Cval ) ); 
        p_WM=p_WM/sum(p_WM);
        p = ns(nStim/3)*p_WM + (1-ns(nStim/3))*p_RL;
        p = (1-epsilon).*p + epsilon*1/3;
      
        if find(stimVec == istim,1,'first') == itrial
            ichoice = subjrespCell{iblock}(itrial);
        else
            ichoice = find([0 cumsum(p)]<rand,1,'last');
        end
        r = (ichoice == corrrespVec(itrial));
        
        if (ichoice ~= -1)
            delta = r-(ns(nStim/3)*imult(icond).*WMvalMat(istim,ichoice) + (1-ns(nStim/3)*imult(icond)).*QvalMat(istim,ichoice));
            % updating
            if (r==1)
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + alpha(icond)*delta;
            else
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + neg_alpha*delta;
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
    QvalCell{iblock} = QvalMat;
end