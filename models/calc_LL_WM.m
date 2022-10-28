function LL = calc_LL_WM(pars,stimvaluesCell,corrCell,responseCell,filler,logflag,fixparams)
%CALC_LL_WM calculates the log-likelihood of data given parameters and data
%
% =========================== INPUT VARIABLES ============================
% PARS: 1x3 vector of parameters [epsilon, lambda, beta]
%       epsilon: lapse rate
%       lambda: WM decay rate
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

if nargin < 5; filler = []; end
if nargin < 6; fixparams = []; end

if sum(logflag); % if there are any logged
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
epsilon = pars(1);
lambda = pars(2);
beta = pars(3);

nBlocks = length(stimvaluesCell);

LL = 0;
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    
    % data
    corrVec = corrCell{iblock};
    responseVec = responseCell{iblock};
    
    llVec = nan(1,nTrials);
    valueMat = ones(nStim,3)./3; % initializing all values for all stim and responses
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        ichoice = responseVec(itrial);
        
        if (ichoice ~= -1)
            p = 1/sum(exp(beta .* (valueMat(istim,:)-valueMat(istim,ichoice))));
            p = (1-epsilon).*p + epsilon*1/3;
            llVec(itrial) = min(max(p,1e-5),1-1e-5);
        end
        
        valueMat = (1-lambda).*1/3 + lambda.*valueMat; % decay
        if (ichoice ~= -1); valueMat(istim,ichoice) = corrVec(itrial); end % current value
    end
    
    LL = LL + nansum(log(llVec));
end