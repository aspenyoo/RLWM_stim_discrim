function [respCell, correctCell] = simulatetest_RLWMi_dn_pers(pars,QvalCell,test_fullseq_learnblocknum,test_fullseq_stimvalues,test_fullseq_corrresp,condVec)
%SIMULATETEST_RL3WM_PERS simulates data given parameters and stimuli for
%test phase
% 
% ========================= INPUT VARIABLES ============================
% PARS: 1x12 vector of parameters 
%    [alpha, neg_alpha, epsilon, lambda, dn_e, dn_c, dn_t ns3, ns6, pers, tau, beta]
%       alpha_e: RL learning rate for positive feedback
%       neg_alpha: RL learning rate for negative feedback
%       epsilon: lapse rate
%       lambda: WM decay rate
%       dn_e: decision noise. how confused observer gets when computing
%       p_RL for exemplar condition
%       dn_c: decision noise for category condition
%       dn_t: decision noise. for text condition
%       ns3: WM (vs RL) weight for set size 3 blocks
%       ns6: WM weight for set size 6 blocks
%       pers: perseveration
%       tau: perseveration decay rate
%       beta_test: softmax inverse temperature parameter of test phase
% QVALCELL: cell of length nBlocks (number of blocks)
%       each cell containing a nStimulus x nActions matrix of Q values
% TEST_LEARNBLOCKNUM: length nTestTrials vector of integers
%       corresponding to the learning block number in which image was
%       presented
% TEST_STIMVALUES: length nTestTrials vector of integers
%       corresponding to the stimulus number of corresponding block
% TEST_FULLSEQ_CORRRESP: length nTestTrials vector of integers
%       corresponding to the correct response of trial
% STIMVALUESCELL: cell of length nBlocks 
%       each cell contains a vector of length nTrials of scalars 
%       corresponding to the index of the stimulus presented on each trial
% CORRRESPCELL: cell of length nBlocks (number of blocks)
%       each cell contains a vector of lanth nTrials of scalars
%       corresponding to the index of the correct button response
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

% pars
epsilon = pars(3);
dnVec = pars(5:7);
pers = pars(10);
tau = pars(11);
beta_test = pars(end-1);
    
% simulate test answers
nTrials = length(test_fullseq_learnblocknum);
Cval = ones(1,3)./1/3;
[respVec, correctVec] = deal(nan(1,nTrials));
for itrial = 1:nTrials
    iblock = test_fullseq_learnblocknum(itrial);
    istim = test_fullseq_stimvalues(itrial);
    corrresp = test_fullseq_corrresp(itrial);
   
    % dn parameter
    icond = condVec(iblock);
    dn = dnVec(icond);
    
    %
    qvalmat = QvalCell{iblock}(istim,:);
    qvalmat = (1-dn).*QvalCell{iblock}(istim,:) + dn*1/3;
    p_RL = exp( beta_test*( qvalmat- pers.*Cval ) );
    
    p_RL = p_RL/sum(p_RL);
    p = p_RL;
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

% for iblock = 1:nBlocks
%     stimVec = test_stimvaluesCell{iblock};
%     nTrials = length(stimVec);
%     icond = condVec(iblock);
%     
%     dn = dnVec(icond);
%     
%     % data
%     corrrespVec = test_corrrespCell{iblock};
%     
%     QvalMat = QvalCell{iblock};
%     Cval = ones(1,3)./1/3;
%     [respVec,correctVec] = deal(nan(1,nTrials));
%     for itrial = 1:nTrials
%         istim = stimVec(itrial);
%         
%         % p of each choice
% %         stimvec = 1:nStim;
% %         stimvec(istim) = [];
% %         qvalmat = (1-dn)*QvalMat(istim,:) + dn*mean(QvalMat(stimvec,:)); % q values used in decision a weighted mean of correct and incorrect states
%         qvalmat = QvalMat(istim,:);
%         p_RL = exp( beta_RL*( qvalmat- pers.*Cval ) );
% 
%         p_RL=p_RL/sum(p_RL);
%         p = p_RL;
%         p = (1-epsilon).*p + epsilon*1/3;
%       
% %         if find(stimVec == istim,1,'first') == itrial
% %             ichoice = subjrespCell{iblock}(itrial);
% %         else
%             ichoice = find([0 cumsum(p)]<rand,1,'last');
% %         end
%         r = (ichoice == corrrespVec(itrial));
%         
%         if (ichoice ~= -1)      
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