function LL = calc_LL_RL_pers(pars,stimvaluesCell,corrCell,responseCell,filler,logflag,fixparams)
%CALC_LL_WM calculates the log-likelihood of data given parameters and data
%
% =========================== INPUT VARIABLES ============================
% PARS: 1x4 vector of parameters [alpha, epsilon, beta]
%       alpha: RL learning rate for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       epsilon: lapse rate
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
pers = pars(4);
tau = pars(5);
beta = pars(end);

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
    Cval = ones(1,3)./1/3;
    QvalMat = ones(nStim,3)./3; % initializing all values for all stim and responses
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        ichoice = responseVec(itrial);
        
        if (ichoice ~= -1)
%             p = 1/sum(exp(beta .* (QvalMat(istim,:)-QvalMat(istim,ichoice))));
            p = 1/sum(exp( beta.*( QvalMat(istim,:)-QvalMat(istim,ichoice) - pers.*(Cval-Cval(ichoice)) ) ) );

            p = (1-epsilon).*p + epsilon*1/3;
            llVec(itrial) = min(max(p,1e-5),1-1e-5);
            
            % updating RL part of value
            if (corrVec(itrial)==1);
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + alpha*(corrVec(itrial)-QvalMat(istim,ichoice));
            else
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + neg_alpha*(corrVec(itrial)-QvalMat(istim,ichoice));
            end
            
            % updating choice trace (perseveration)
            delta_choice = 0-Cval;
            delta_choice(ichoice) = 1-Cval(ichoice);
            Cval = Cval + tau.*delta_choice;
        end
    end
    
    LL = LL + nansum(log(llVec));
end