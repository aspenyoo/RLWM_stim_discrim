function [respCell, correctCell] = simulatetest_RL3WMi_pers(pars,QvalCell,test_fullseq_learnblocknum,test_fullseq_stimvalues,test_fullseq_corrresp,condVec)
%SIMULATETEST_RL3WM_PERS simulates data given parameters and stimuli for
%test phase
% 
% ========================= INPUT VARIABLES ============================
%       epsilon: lapse rate
%       pers: perseveration
%       tau: perseveration decay rate
%       beta: softmax inverse temperature parameter
% QVALCELL: cell of length nBlocks (number of blocks)
%       each cell containing a nStimulus x nActions matrix of Q values
% TEST_FULLSEQ_LEARNBLOCKNUM: 1xnTestTrials vector of integers
%       corresponding to the learning block number in which image was
%       presented
% TEST_FULLSEQ_STIMVALUES: 1xnTestTrials vector of integers
%       corresponding to the stimulus number of corresponding block
% TEST_FULLSEQ_CORRRESP: 1xnTestTrials vector of integers
%       corresponding to the correct response of trial stimulus
% CONDVEC: 1 x nBlocks vector, condition indices 
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
% written by aspen yoo, dec 2021
% aspen.yoo@berkeley.edu


% % pars
% alpha = pars(1:3);
% neg_alpha = pars(4);
epsilon = pars(5);
% lambda = pars(6);
% ns = pars(7:8);
pers = pars(9);
tau = pars(10);
beta_test = pars(11);
% beta_WM = 100;

% simulate test answers
nTrials = length(test_fullseq_learnblocknum);
Cval = ones(1,3)./1/3;
[respVec, correctVec] = deal(nan(1,nTrials));
for itrial = 1:nTrials
    iblock = test_fullseq_learnblocknum(itrial);
    istim = test_fullseq_stimvalues(itrial);
    corrresp = test_fullseq_corrresp(itrial);
    
    p_RL = exp( beta_test*( QvalCell{iblock}(istim,:) - pers.*Cval ) );
    p = p_RL/sum(p_RL);     
    p = (1-epsilon).*p + epsilon*1/3;
    
    ichoice = find([0 cumsum(p)]<rand,1,'last');
    r = (ichoice == corrresp);
    
    % updating choice trace (perseveration)
    delta_choice = 0-Cval;
    delta_choice(ichoice) = 1-Cval(ichoice);
    Cval = Cval + tau.*delta_choice;
    
    % saving variables
    respVec(itrial) = ichoice;
    correctVec(itrial) = r;
end

% create respCell and correctCell, in desired format
[respCell, correctCell] = deal(cell(1,12));
for iblock = 1:12
    
    idx = (test_fullseq_learnblocknum == iblock); % which trials are in current block
    
    respCell{iblock} = respVec(idx);
    correctCell{iblock} = correctVec(idx);
end



% 
% 
% 
% nBlocks = length(test_stimvaluesCell);
% 
% [respCell, correctCell] = deal(cell(1,nBlocks));
% for iblock = 1:nBlocks
%     stimVec = test_stimvaluesCell{iblock};
%     nTrials = length(stimVec);
%     nStim = max(stimVec);
%     icond = condVec(iblock);
%     
%     % data
%     corrrespVec = test_corrrespCell{iblock};
%     
%     Cval = ones(1,3)./1/3;
%     [respVec,correctVec] = deal(nan(1,nTrials));
%     QvalMat = QvalCell{iblock};
% 
% %     [QvalMat,WMvalMat] = deal(ones(nStim,3)./3); % initializing all values for all stim and responses    
%     for itrial = 1:nTrials
%         istim = stimVec(itrial);
%         
%         % p of each choice 
%         p_RL = exp( beta_RL*( QvalMat(istim,:) - pers.*Cval ) );
%         p_RL=p_RL/sum(p_RL);
% %         p_WM = exp( beta_WM*( WMvalMat(istim,:) - pers.*Cval ) );
% %         p_WM = p_WM/sum(p_WM);
%         p = p_RL;
% %         p = ns(nStim/3)*p_WM + (1-ns(nStim/3))*p_RL;
%         p = (1-epsilon).*p + epsilon*1/3;
%       
% %         if find(stimVec == istim,1,'first') == itrial
% %             ichoice = test_subjrespCell{iblock}(itrial);
% %         else
%             ichoice = find([0 cumsum(p)]<rand,1,'last');
% %         end
%         r = (ichoice == corrrespVec(itrial));
%         
%         if (ichoice ~= -1)
%             
%             % updating choice trace (perseveration)
%             delta_choice = 0-Cval; 
%             delta_choice(ichoice) = 1-Cval(ichoice); 
%             Cval = Cval + tau.*delta_choice; 
%         end
%         
%         % saving variables
%         respVec(itrial) = ichoice; 
%         correctVec(itrial) = r; 
%     end
%     
%     respCell{iblock} = respVec; 
%     correctCell{iblock} = correctVec; 
% end