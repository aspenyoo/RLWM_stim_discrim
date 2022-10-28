function [respCell, correctCell] = simulate_superfree(model,pars,stimvaluesCell,corrrespCell,condVec,subjrespCell)
%SIMULATE_SUPERFREE simulates data given parameters and stimuli
% 
% ========================= INPUT VARIABLES ============================
% PARS: 1x18 vector of parameters
%    [alpha, neg_alpha, epsilon, lambda, ns3, ns6] for each condition
%       alpha: RL learning rate for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       epsilon: lapse rate
%       lambda: WM decay rate
%       ns3: WM (vs RL) weight for set size 3 blocks
%       ns6: WM weight for set size 6 blocks
%       beta: softmax inverse temperature parameter (fixed to 100)
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
% written by aspen yoo, march 2021
% aspen.yoo@berkeley.edu

nBlocks = length(condVec);
[respCell,correctCell] = deal(cell(1,nBlocks));

[LB,~,~,~,~,~,~,~,~,~,fixparams] = loadfittingparams(model);
n = length(LB);

if strcmp(model(end-1:end),'_0')
    modell = model(1:end-2);
else
    modell = model;
end
try % if fitting no dn/ca parameter for category condtiion
    if strcmp(modell(end-3:end),'_sub')
        modell = modell(1:end-4);
    end
end
try % if fitting tau as free parameter
    if strcmp(modell(end-7:end-4),'full')
        modell = [modell(1:end-8) 'pers'];
    end
end

nConds = length(unique(condVec));
for icond = 1:nConds
    idx_block = condVec == icond; % blocks of current condition
    
    % get subset of data
    svCell = stimvaluesCell(idx_block);
    crCell = corrrespCell(idx_block);
    cVec = condVec(idx_block);
    srVec = subjrespCell(idx_block);
    
%     theta = pars((icond-1)*6+(1:6));
    
    idx = ((icond-1)*n+1):(icond*n);
    theta = pars(idx);
    
    % if there are fixed parameters
    if ~isempty(fixparams)
        nParams = length(theta) + size(fixparams,2);
        nonfixedparamidx = 1:nParams;
        nonfixedparamidx(fixparams(1,:)) = [];
        
        temptheta = nan(1,nParams);
        temptheta(nonfixedparamidx) = theta;
        temptheta(fixparams(1,:)) = fixparams(2,:);
        
        theta = temptheta;
    end

    eval(sprintf('[respCell(idx_block), correctCell(idx_block)] = simulate_%s(theta,svCell,crCell,cVec,srVec);',modell));

end
