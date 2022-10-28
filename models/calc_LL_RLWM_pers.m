function LL = calc_LL_RLWM_pers(pars,stimvaluesCell,corrCell,responseCell,condVec,logflag,fixparams)
%CALC_LL_RLWM calculates the log-likelihood of data given parameters for
%RLWM model
%
% =========================== INPUT VARIABLES ============================
% PARS: 1x7 vector of parameters [alpha, epsilon, lambda, ns3, ns6, beta]
%       alpha: RL learning rate for positive feedback
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
% CORRCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to whether participant was rewarded
% RESPONSECELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to the participant's response
% CONDVEC: 1 x nBlocks vector, condition indices (not used in this function)
% LOGFLAG: vector of length number of parameters
%       each element a logical whether parameter is logged (and thus needs
%       to be exponentiated)
% FIXPARAMS: (optional). 2 x (number of fixed parameters) matrix. fixed 
%     parameters, such that the first row corresponds to the index and 
%     second row corresponds to the value of the fixed parameter. 
%
% ========================= OUTPUT VARIABLES ============================
% LL: scalar, log likelihood of data given model and parameters
%
% written by aspen yoo, november 2020
% aspen.yoo@berkeley.edu

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
epsilon = pars(3);
lambda = pars(4);
ns = pars(5:6);
pers = pars(7);
tau = pars(8);
beta_RL = pars(end);
beta_WM = 100;

nBlocks = length(stimvaluesCell);

LL = 0;
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    
    % data
    corrVec = corrCell{iblock};
    responseVec = responseCell{iblock};
    
    Cval = ones(1,3)./1/3;
    llVec = nan(1,nTrials);
    [QvalMat,WMvalMat] = deal(ones(nStim,3)./3); % initializing all values for all stim and responses
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        ichoice = responseVec(itrial);
        
        
        if (ichoice ~= -1)
            % RL part
            p_RL = 1/sum(exp( beta_RL.*( QvalMat(istim,:)-QvalMat(istim,ichoice) - pers.*(Cval-Cval(ichoice)) ) ) );
            
            % WM part
            p_WM = 1/sum(exp( beta_WM.*( WMvalMat(istim,:)-WMvalMat(istim,ichoice) - pers.*(Cval-Cval(ichoice))) ) );
            
            p = ns(nStim/3)*p_WM + (1-ns(nStim/3))*p_RL; 
            p = (1-epsilon).*p + epsilon*1/3;
            
            llVec(itrial) = min(max(p,1e-5),1-1e-5);

            % updating value
            if (corrVec(itrial)==1)
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + alpha*(corrVec(itrial)-QvalMat(istim,ichoice));
            else
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + neg_alpha*(corrVec(itrial)-QvalMat(istim,ichoice));
            end
            
            % updating choice trace (perseveration)
            delta_choice = 0-Cval;
            delta_choice(ichoice) = 1-Cval(ichoice);
            Cval = Cval + tau.*delta_choice;
        end
        
        % updating WM part of value
        WMvalMat = (1-lambda).*1/3 + lambda.*WMvalMat; % delay
        if (ichoice ~= -1); WMvalMat(istim,ichoice) = corrVec(itrial); end
        
    end
        LL = LL + nansum(log(llVec));
end