function [LL, QvalCell] = calc_LL_RLWMi_dn_all_pers(pars,stimvaluesCell,corrCell,responseCell,test_learnblocknum,test_stimvalues,test_subjresp,condVec,logflag,fixparams)
%CALC_LL_RLWM_dn calculates the log-likelihood of data given parameters for
%RL3WM model
%
% =========================== INPUT VARIABLES ============================
% PARS: 1x13 vector of parameters 
%    [alpha, neg_alpha, epsilon, lambda, dn_e, dn_c, dn_t ns3, ns6, pers, tau, beta]
%       alpha_e: RL learning rate for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       epsilon: lapse rate
%       lambda: WM decay rate
%       dn_e: decision noise. how confused observer gets when computing
%       p_RL for exemplar condition
%       dn_c: decision noise category coniditon
%       dn_t: decision noise. for text condition
%       ns3: WM (vs RL) weight for set size 3 blocks
%       ns6: WM weight for set size 6 blocks
%       pers: perseveration
%       tau: perseveration decay rate
%       beta_test: softmax inverse temperature parameter for test phase
%       beta: softmax inverse temperature parameter for learning phase
% STIMVALUESCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars 
%       corresponding to the index of the stimulus presented on each trial
% CORRCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to whether participant was rewarded
% RESPONSECELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of length nTrials of scalars
%       corresponding to the participant's response
% TEST_LEARNBLOCKNUM: length nTestTrials vector of integers
%       corresponding to the learning block number in which image was
%       presented
% TEST_STIMVALUES: length nTestTrials vector of integers
%       corresponding to the stimulus number of corresponding block
% TEST_SUBJRESP: length nTestTrials vector of integers
%       corresponding to the participant's response of trial
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
% values after learning block
%
% written by aspen yoo, november 2020
% aspen.yoo@berkeley.edu


if nargin < 10; fixparams = []; end
if nargin < 9; logflag = []; end

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
dnVec = pars(5:7);
ns = pars(8:9);
pers = pars(10);
tau = pars(11);
beta_test = pars(12);
beta_RL = pars(13);
beta_WM = 100;

nBlocks = length(stimvaluesCell);

% ======== LEARNING PHASE LL ======
LL = 0;
QvalCell = cell(1,nBlocks);
for iblock = 1:nBlocks
    stimVec = stimvaluesCell{iblock};
    nTrials = length(stimVec);
    nStim = max(stimVec);
    icond = condVec(iblock);
    
    dn = dnVec(icond);
            
    % data
    corrVec = corrCell{iblock};
    responseVec = responseCell{iblock};
    
    llVec = nan(1,nTrials);
    Cval = ones(1,3)./1/3;
    [QvalMat,WMvalMat,] = deal(ones(nStim,3)./3); % initializing all values for all stim and responses
    for itrial = 1:nTrials
        istim = stimVec(itrial);
        ichoice = responseVec(itrial);
        
        if (ichoice ~= -1)
            % RL part
            stimvec = 1:nStim;
            stimvec(istim) = [];
            qvalmat = (1-dn)*QvalMat(istim,:) + dn*mean(QvalMat(stimvec,:)); % q values used in decision a weighted mean of correct and incorrect states
            p_RL = 1/sum(exp( beta_RL.*( qvalmat-qvalmat(ichoice) - pers.*(Cval-Cval(ichoice))) ));
            
            % WM part
            p_WM = 1/sum(exp( beta_WM.*(WMvalMat(istim,:)-WMvalMat(istim,ichoice) - pers.*(Cval-Cval(ichoice))) ));
            
            p = ns(nStim/3)*p_WM + (1-ns(nStim/3))*p_RL; 
            p = (1-epsilon).*p + epsilon*1/3;
            
            llVec(itrial) = min(max(p,1e-5),1-1e-5);
            
            % updating RL part of value
            delta = corrVec(itrial)-(ns(nStim/3).*WMvalMat(istim,ichoice) + (1-ns(nStim/3)).*QvalMat(istim,ichoice));
            if (corrVec(itrial)==1)
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + alpha*delta;
            else
                QvalMat(istim,ichoice) = QvalMat(istim,ichoice) + neg_alpha*delta;
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

% ====== TEST PHASE LL =====
nTestTrials = length(test_learnblocknum);
llVec = nan(1,nTestTrials);
for itrial = 1:nTestTrials
    
    iblock = test_learnblocknum(itrial);
    istim = test_stimvalues(itrial);
    subjresp = test_subjresp(itrial);
    
    if (subjresp ~= -1)
        stimvec = 1:nStim;
        stimvec(istim) = [];
        qvalmat = (1-dn)*QvalCell{iblock}(istim,:) + dn*mean(QvalCell{iblock});
        p = 1/sum(exp( beta_test.*( qvalmat-qvalmat(subjresp) - pers.*(Cval-Cval(subjresp))) ));
        p = (1-epsilon).*p + epsilon*1/3;
        
        llVec(itrial) = min(max(p,1e-5),1-1e-5);
        
        % updating choice trace (perseveration)
        delta_choice = 0-Cval;
        delta_choice(subjresp) = 1-Cval(subjresp);
        Cval = Cval + tau.*delta_choice;
    end   
end

LL = LL + nansum(log(llVec));