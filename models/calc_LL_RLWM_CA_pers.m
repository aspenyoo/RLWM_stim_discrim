function [LL, QvalCell] = calc_LL_RLWM_CA_pers(pars,stimvaluesCell,corrCell,responseCell,condVec,logflag,fixparams)
%CALC_LL_RLWM_CA calculates the log-likelihood of data given parameters for
%RL3WM model
%
% =========================== INPUT VARIABLES ============================
% PARS: 1x11 vector of parameters 
%    [alpha, neg_alpha, CA_e, CA_t ,epsilon, lambda, ns3, ns6, pers, tau, beta]
%       alpha: RL learning rate for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       CA: credit assignment confusion parameter (for exemplar and text
%       conditions only)
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
% CORRCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to whether participant was rewarded
% RESPONSECELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to the participant's response
% CONDVEC: 1 x nBlocks vector, condition indices 
% LOGFLAG: vector of length number of parameters
%       each element a logical whether parameter is logged (and thus needs
%       to be exponentiated)
% FIXPARAMS: (optional). 2 x (number of fixed parameters) matrix. fixed 
%     parameters, such that the first row corresponds to the index and 
%     second row corresponds to the value of the fixed parameter. 
%
% ========================= OUTPUT VARIABLES ============================
% LL: scalar, log likelihood of data given model and parameters
% QVALCELL: 1xnBlocks cell. each containing nSxnResponses matrix of Q
%       values after learning block
%
% written by aspen yoo, november 2020
% aspen.yoo@berkeley.edu


if nargin < 7; fixparams = []; end
if nargin < 6; logflag = []; end

if sum(logflag) % if there are any logged
    pars(logflag) = exp(pars(logflag));
end

% if there are fixed parameters
if ~isempty(fixparams)
    nParams = length(pars) + size(fixparams,2);
    nonfixedparamidx = 1:nParams;
    nonfixedparamidx(fixparams(1,:)) = [];
    
    temptheta = nan(1,nParams);
    temptheta(nonfixedparamidx) = pars;
    temptheta(fixparams(1,:)) = fixparams(2,:);
    
    pars = temptheta;
end

% pars
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

LL = 0;
QvalCell = cell(1,nBlocks);
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
    corrVec = corrCell{iblock};
    responseVec = responseCell{iblock};
    
    llVec = nan(1,nTrials);
    Cval = ones(1,3)./1/3;
    [QvalMat,WMvalMat] = deal(ones(nStim,3)./3); % initializing all values for all stim and responses
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        ichoice = responseVec(itrial);
        
        if (ichoice ~= -1)
            % RL part
            p_RL = 1/sum(exp( beta_RL.*( QvalMat(istim,:)-QvalMat(istim,ichoice) - pers.*(Cval-Cval(ichoice)))));
            
            % WM part
            p_WM = 1/sum(exp( beta_WM.*(WMvalMat(istim,:)-WMvalMat(istim,ichoice) - pers.*(Cval-Cval(ichoice)) )));
            
            p = ns(nStim/3)*p_WM + (1-ns(nStim/3))*p_RL; 
            p = (1-epsilon).*p + epsilon*1/3;
            
            llVec(itrial) = min(max(p,1e-5),1-1e-5);
            
            % updating RL part of value
            stimvec = 1:nStim;
            stimvec(istim) = [];
            delta = corrVec(itrial)-QvalMat(istim,ichoice);
            if (corrVec(itrial)==1)
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + alpha*delta;
                QvalMat(stimvec,ichoice) = QvalMat(stimvec,ichoice) + alpha*CA*delta; % update other states because you're confused which state you're in
            else
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + neg_alpha*delta;
                QvalMat(stimvec,ichoice) = QvalMat(stimvec,ichoice) + neg_alpha*CA*delta;
            end
            
            % updating choice trace (perseveration)
            delta_choice = 0-Cval;
            delta_choice(ichoice) = 1-Cval(ichoice);
            Cval = Cval + tau.*delta_choice;
        end
        
        % updating WM part of value
        WMvalMat = (1-lambda).*1/3 + lambda.*WMvalMat; % decay
        if (ichoice ~= -1); WMvalMat(istim,ichoice) = corrVec(itrial); end
    end
    
    QvalCell{iblock} = QvalMat;
    LL = LL + nansum(log(llVec));
end