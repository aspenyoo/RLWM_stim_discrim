%% ================================================================
%                     BEHAVIORAL ANALYSES
%  ================================================================
% setup
addpath('helper_functions')
addpath('models')

%% MAKE CSV FOR ANOVA (which is done in R)

clear all

load('experimentalsettings.mat')
exptype = 'RPP';

nSubjs = nSubjs.(exptype);
nreps = 11;

[PC, CONDITION, SETSIZE, SUBJECT] = deal([]);
for isubj = 1:nSubjs
     subjid = subjidVec.(exptype)(isubj);
    
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    for icond = 1:3 % picture conditions
        idx_condblock = condVec == conditionMat(2,icond);
        
        for ns = [3 6]
            idx_block = find(idx_condblock & (nStimsVec == ns));
            
            temppc = [];
            for iblock = idx_block
                % remove nonresponses
                idx_keep = corrCell{iblock} ~= -1;
                sv = stimvaluesCell{iblock}(idx_keep);
                corr = corrCell{iblock}(idx_keep);
                
                if strcmp('RPP',exptype) && (isubj == 28) && (iblock == 6)
                else
                for istim = 1:ns
                    idx = find(sv == istim);
                    temppc = [temppc; nanmean(corr(idx))];
                end
                end
            end
            
            PC = [PC; nanmean(temppc)];
            CONDITION = [CONDITION; icond];
            SETSIZE = [SETSIZE; ns];
            SUBJECT = [SUBJECT; isubj];
        end
    end
end

tbl = table(PC(:), CONDITION(:),SETSIZE(:),SUBJECT(:),'VariableNames',{'PC','CONDITION','SETSIZE','SUBJECT'});

% write table to csv for R
writetable(tbl,sprintf('dataforanova_%s.csv',exptype));

%% training: logistic regression
% p(correct) ~ nstims + 1/delay since last correct + number of correct

% see plot_figures.m for FIGURE 1D & 3A(right)


%% test phase: descriptive statistics
% is test phase performance better than chance?
% is there a tortoise hare effect for each condition?

clear all

load('experimentalsettings.mat')
exptype = 'RPP';
nBlocks = 12;
         
condVec = nan(nSubjs.(exptype),nBlocks); %
traincorrMat1 = cell(1,nConds); %nan(nSubjs,nTimes);
testcorrMat1 = cell(1,nConds);
traincorrMat = cell(1,nConds); %nan(nSubjs,nTimes);
testcorrMat = cell(1,nConds);

for isubj = 1:nSubjs.(exptype)
    subjid = subjidVec.(exptype)(isubj);

    % ======= NEW CODE ========
    % load data
    load(sprintf('data/%s/fittingdata_subjid%d.mat%',exptype,subjid))
    testcorrVec = double(test_fullseq_corrresp == test_fullseq_subjresp);
    
    [traincorrmat, testcorrmat] = deal(cell(1,nConds));
    for iblock = 1:nBlocks
        
        % condition information
        nStims = nStimsVec(iblock);
        condition = condVec(iblock);
        icond = find(conditionMat(1,:)==nStims & conditionMat(2,:)==condition);
        
        % training accuracy info
        iscorrVec = corrCell{iblock}; % correct or not
        
        % testing accuracy info
        idx = (test_fullseq_learnblocknum == iblock); % which trials are in current block
        
        for istim = 1:nStims
            % training
            corrvec = iscorrVec(stimvaluesCell{iblock} == istim); % all relevant stim times
            traincorrmat{icond} = [traincorrmat{icond}; corrvec(end-2:end)];
            
            % testing
            idxx = idx & (test_fullseq_stimvalues == istim);
            testcorrmat{icond} = [testcorrmat{icond}; testcorrVec(idxx)'];
        end
    end
    
    for icond = 1:nConds
        % training
        traincorrMat{icond}(isubj,:) = nanmean(traincorrmat{icond}(traincorrmat{icond} ~= -1));
        
        % testing
        testcorrMat{icond}(isubj,:) = nanmean(testcorrmat{icond}(testcorrmat{icond} ~= -1));
    end
end

% is test phase perfomrance better than chance?
[h,p,ci,stats] = ttest(cell2mat(testcorrMat)-1/3);
stats.tstat
p

% is there a tortoise hare effect?
testtraindiff = cellfun(@minus,traincorrMat,testcorrMat,'UniformOutput',false);

tortoisehare = cell2mat(testtraindiff);
tortoisehare = tortoisehare(:,1:3) - tortoisehare(:,4:6);

conditionmat = repmat(1:3,nSubjs.(exptype),1);
% setsizemat = repmat(conditionMat(1,:),nSubjs,1);
subjidmat = repmat([1:nSubjs.(exptype)]',1,nConds/2);
tbl = table(tortoisehare(:), conditionmat(:),subjidmat(:),'VariableNames',{'TH','condition','subject'});

% write table to csv for R
writetable(tbl,'dataforanova_tortoisehare.csv');

% anova can be run in "TORTOISE AND HARE section" of the file "anova.R"


%% 

for icond = 1:6
    
    sum(traincorrMat{icond} ~= traincorrMat{icond})
%     sum(testcorrMat{icond} ~= testcorrMat1{icond})
    
end
%% ================================================================
%                           FIT MODELS
%  ================================================================

%% EXP 1 AND 2: FIT PARAMETERS TO LEARNING PHASE DATA

clear all

exptype = 'Mturk'; % 'RPP';
% model = 'RL_pers_0';
% model = 'WM_pers_0';
model = 'RLWM_pers_0';

if strcmp(model(end-1:end),'_0') % fixing \alpha_- = 0
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

% load experimental and fitting settings
load('experimentalsettings.mat')
nReps = 20;
[LB,UB,PLB,PUB,logflag,A,b,Aeq,beq,nonlcon,fixparams] = loadfittingparams(model);
n = length(PLB);

for isubj = 1:nSubjs.(exptype)
    isubj

    subjid = subjidVec.(exptype)(isubj);
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    x0_list = latinhs(nReps,n,PLB,PUB,[],1e3);
    xbestmat = nan(nReps,n);
    LLvec = nan(1,nReps);
    
    fun = eval(sprintf('@(x) -calc_LL_%s(x,stimvaluesCell,corrCell,responseCell,condVec,logflag,fixparams)',modell));
    
    for irep = 1:nReps
        x0 = x0_list(irep,:);
        [xbestmat(irep,:),LLvec(irep),~,~] = fmincon(fun,x0,A,b,Aeq,beq,LB,UB,nonlcon);
    end
    
    % ----- save to file -----
    filename = sprintf('fits/%s/%s/subjid%d.mat',exptype,model,subjid); % filename
    try
        tempx = xbestmat;
        tempx(:,logflag) = exp(tempx(:,logflag));
        templl = LLvec;
        load(filename,'xbestmat','LLvec');
        xbestmat = [xbestmat; tempx];
        LLvec = [LLvec templl];
    catch
        xbestmat(:,logflag) = exp(xbestmat(:,logflag));
    end
    
    % save
    if ~exist(sprintf('fits/%s/%s',exptype,model), 'dir')
       mkdir(sprintf('fits/%s/%s',exptype,model))
    end
    save(filename,'xbestmat','LLvec')
end

%% EXP 1, 2, RECOVERY: FIT SUPERFREE MODEL PARAMETERS

clear all

load('experimentalsettings.mat')
exptype = 'Mturk'; % 'Mturk (exp1), 'RPP' (exp2), 'recovery' (parameter/model recovery)

model = 'dnRLWM_pers_0';

if strcmp(exptype,'recovery')
    nSubjs = 50;
    simmodel = 'RL3WM3_pers_0'; % model that simulated data to be fit
else
    nSubjs = nSubjs.(exptype);
end


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

nReps = 20;
[LB,UB,PLB,PUB,logflag,A,b,Aeq,beq,nonlcon,fixparams] = loadfittingparams(model);
n = length(PLB);
lf = repmat(logflag,1,3);

for isubj = 1:nSubjs
    isubj
    
    if strcmp(exptype,'recovery')
        load(sprintf('data/simdata/simdata_model_%s_subj%d.mat',simmodel,isubj))
    else
        subjid = subjidVec.(exptype)(isubj);
        load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    end
    
    x0_list = latinhs(nReps,n,PLB,PUB,[],1e3);
    for irep = 1:nReps
        x0 = x0_list(irep,:);
        
        xbest = []; LL = 0;
        for icond = 1:3
            idx_block = condVec == icond; % blocks of current condition
            
            % get subset of data
            svCell = stimvaluesCell(idx_block);
            cCell = corrCell(idx_block);
            rCell = responseCell(idx_block);
            cVec = condVec(idx_block);
            
            eval(sprintf('[xb,ll,~,~] = fmincon(@(x) -calc_LL_%s(x,svCell,cCell,rCell,cVec,logflag,fixparams),x0,A,b,Aeq,beq,LB,UB);',modell));
           
            xbest = [xbest xb];
            LL = LL+ll;
        end
        xbestmat(irep,:) = xbest;
        LLvec(irep) = LL;
    end
   
    xbestmat(:,lf) = exp(xbestmat(:,lf));
    
    if strcmp(exptype,'recovery')
        if ~exist('fits/modelrecovery', 'dir')
            mkdir('fits/modelrecovery')
        end
        save(sprintf('fits/modelrecovery/fitmodel_superfree_%s_simmodel_%s_subj%d.mat',model,simmodel,isubj),'xbestmat','LLvec');
        
    else
        if ~exist(sprintf('fits/%s/superfree_%s',exptype,model), 'dir')
            mkdir(sprintf('fits/%s/superfree_%s',exptype,model))
        end
        save(sprintf('fits/%s/superfree_%s/subjid%d.mat',exptype,model,subjid),'xbestmat','LLvec')
    end
end

%% EXP 2: FIT PARAMETERS OF RPP PARTICIPICIPANTS TO TRAIN AND TEST DATA

clear all

exptype = 'RPP';
model = 'RL3WMi_all_pers_0';
% model = 'RLWMi_dn_all_pers_sub_0';


if strcmp(model(end-1:end),'_0') % if fixing \alpha_- = 0
    modell = model(1:end-2);
else
    modell = model;
end
if strcmp(modell(end-3:end),'_sub') % if not fitting dn/ca parameter for category condtiion
    modell = modell(1:end-4);
end

% load experimental and fitting settings
load('experimentalsettings.mat')
nReps = 20;
[LB,UB,PLB,PUB,logflag,A,b,Aeq,beq,nonlcon,fixparams] = loadfittingparams(model);
n = length(PLB);

for isubj = 1:nSubjs.(exptype)
    isubj
    
    subjid = subjidVec.(exptype)(isubj);
    load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjid))
    
    x0_list = latinhs(nReps,n,PLB,PUB,[],1e3);
    xbestmat = nan(nReps,n);
    LLvec = nan(1,nReps);
    
    fun = eval(sprintf('@(x) -calc_LL_%s(x,stimvaluesCell,corrCell,responseCell,test_fullseq_learnblocknum,test_fullseq_stimvalues,test_fullseq_subjresp,condVec,logflag,fixparams)',modell));

    for irep = 1:nReps
        x0 = x0_list(irep,:);
        [xbestmat(irep,:),LLvec(irep),~,~] = fmincon(fun,x0,A,b,Aeq,beq,LB,UB,nonlcon);
    end
    
    % ----- save to file ------
    filename = sprintf('fits/%s/%s/subjid%d.mat',exptype,model,subjid); % filename
    try
        tempx = xbestmat;
        tempx(:,logflag) = exp(tempx(:,logflag));
        templl = LLvec;
        load(filename,'xbestmat','LLvec');
        xbestmat = [xbestmat; tempx];
        LLvec = [LLvec templl];
    catch
        xbestmat(:,logflag) = exp(xbestmat(:,logflag));
    end
    
    % save
    if ~exist(sprintf('fits/%s/%s',exptype,model), 'dir')
       mkdir(sprintf('fits/%s/%s',exptype,model))
    end
    save(filename,'xbestmat','LLvec')
end

%% combine fits across participants

clear all

exptype = 'Mturk'; % 'Mturk' or 'RPP'
model = 'RLWM_pers_0';
% model = 'Decay3_fullpers_0'; % model name

load('experimentalsettings.mat')

for isubj = 1:nSubjs.(exptype)
    subjid = subjidVec.(exptype)(isubj);
    
    load(sprintf('fits/%s/%s/subjid%d.mat',exptype,model,subjid),'xbestmat','LLvec');
    
    idx = find(LLvec == min(LLvec),1,'first');
    xbest = xbestmat(idx,:);
    LL = LLvec(idx);
    
    xbestMat(isubj,:) = xbest;
    LLVec(isubj) = LL;
end

save(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'xbestMat','LLVec')

%% model comparison 

%% ===================================================================
%               INTERPRETTING MODEL PARAMETERS
% ====================================================================


%% comparing parameters of interest in winning models

clear all

model = 'RL3WM_pers_0';
% model = 'RLWM_dn_pers_sub_0';
load('experimentalsettings.mat')
paramnames = paramnamesVec.(model);

switch model
    case 'RL3WM_pers_0'
        paramsofinterest = [1 2; 1 3; 2 3];
    case 'RLWM_dn_pers_sub_0'
        paramsofinterest = [4 5];
end

exptype = 'RPP';
load(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'xbestMat')
xb = xbestMat;

exptype = 'Mturk';
load(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'xbestMat')
xb = [xb; xbestMat];

nComps = size(paramsofinterest,1);
for icomp = 1:nComps
    idxx = paramsofinterest(icomp,:);
    [h,p,~,stats] = ttest2(xb(:,idxx(1)),xb(:,idxx(2)));
    fprintf('mean 1: %2.4f, %2.4f \n', mean(xb(:,idxx(1))), std(xb(:,idxx(1)))/sqrt(size(xb,1)))
    fprintf('mean 2: %2.4f, %2.4f \n', mean(xb(:,idxx(2))), std(xb(:,idxx(2)))/sqrt(size(xb,1)))
    fprintf('%s vs %s: t(%d) = %2.3f, p=%0.3f \n',paramnames{idxx(1)},paramnames{idxx(2)},stats.df, stats.tstat,p.*nComps)
end


%% ===================================================================
%               SCRIPTS FOR SUPPLEMENTARY MATERIALS
% ====================================================================

%% generated simulated data
% using fitted parameters from real data

clear all

nsimSubjs = 50;
load('experimentalsettings.mat')
exptypeCell = {'RPP','Mturk'};
% exptypeCell = {'RPP'};
genfromreal_flag = 1;

modelVec = {'RL3WM_pers_0','Decay3_pers_0','RLWM_dn_pers_sub_0','RLWM_CA_pers_0','RLWM_dnwm_pers_sub_0','ns2_pers_0'};
% modelVec = {'RL3WMi_pers_0','RLWMi_dn_pers_sub_0'};
nModels = length(modelVec);

if ~exist('data/simdata', 'dir')
    mkdir('data/simdata')
end

for imodel = 1:nModels
    model = modelVec{imodel}
    
    % get model title for simulation
    if strcmp(model(end-1:end),'_0')
        modell = model(1:end-2);
        if strcmp(modell(end-3:end),'_sub')
            modell = modell(1:end-4);
        end
    else
        modell = model;
    end
    
    switch model
        case 'RL3WMi_all_pers_0'
            modell = 'RL3WMi_pers';
            all_flag = 1;
        case 'RLWMi_dn_all_pers_sub_0'
            modell = 'RLWMi_dn_pers';
            all_flag = 1;
        otherwise
            all_flag = 0;
    end
    
    [LB,UB,PLB,PUB,logflag,A,b,Aeq,beq,nonlcon,fixparams] = loadfittingparams(model);
    xb = [];
    for iexp = 1:length(exptypeCell)
        exptype = exptypeCell{iexp};
        load(sprintf('fits/%s/fits_model_%s.mat',exptype,model),'xbestMat')
        xb = [xb; xbestMat];
    end
    xbestMat = xb;
    
    if (genfromreal_flag)
        nSubjs = size(xbestMat,1);
        if (nSubjs < nsimSubjs) % if there aren't enough "real" parameters, simulate fake ones
            error('not enough real subjects')
        else
            xbestMat = xbestMat(randperm(nSubjs),:);
        end
    else
        % z score parameters, then get mean and cov matrix
        idx_logit = 1:size(xbestMat,2);
        idx_logit(find(sum(xbestMat<0))) = [];
        xbestMat(:,idx_logit) = log(xbestMat(:,idx_logit))-log(1-xbestMat(:,idx_logit)); % logit transform
        xbest = bsxfun(@minus,xbestMat,mean(xbestMat));
        xbest = bsxfun(@rdivide,xbest,std(xbestMat));
        m = mean(xbest);
        sigma = cov(xbest);
    end
    
    for isubj = 1:nsimSubjs
        load(sprintf('data/%s/fittingdata_subjid%d.mat',exptype,subjidVec.(exptype)(mod(isubj,length(subjidVec.(exptype)))+1)))

        if ~(genfromreal_flag)
            isviolated = 1;
            while (isviolated)
                x = mvnrnd(m,sigma);
                p = normcdf(x);
                for iparam = 1:length(x)
                    
                    [f,xi] = ksdensity(xbestMat(:,iparam));
                    f_cdf = cumsum(f)/sum(f); % get smoothed probalitly dist
                    idx = find(abs(f_cdf-p(iparam))==min(abs(f_cdf-p(iparam))));
                    x(iparam) = xi(idx);
                    
                    isviolated = (LB(iparam) > x(iparam)) || (UB(iparam) < x(iparam));
                end
                try
                    isviolated = (A*x'>=b);
                end
                x(1:idx_logit) = exp(x(1:idx_logit))./(1+exp(x(1:idx_logit)));
            end
        else
            x = xbestMat(isubj,:);
        end
        
        if ~isempty(fixparams)
            nParams = length(x) + size(fixparams,2);
            nonfixedparamidx = 1:nParams;
            nonfixedparamidx(fixparams(1,:)) = [];
            
            temptheta = nan(1,nParams);
            temptheta(nonfixedparamidx) = x;
            temptheta(fixparams(1,:)) = fixparams(2,:);
            
            x = temptheta;
        end
        
        % calc_LL_%s(x,stimvaluesCell,corrCell,responseCell,condVec,fixparams)
        
        eval(sprintf('respCell = simulate_%s(x,stimvaluesCell,corrrespCell,condVec,responseCell);',modell))
        
        responseCell = respCell;
        corrCell = cellfun(@(x,y) x==y, responseCell,corrrespCell,'UniformOutput',false);

        save(sprintf('data/simdata/simdata_model_%s_subj%d.mat',model,isubj),...
            'x','corrCell','responseCell','corrrespCell',...
            'stimvaluesCell','condVec','nStimsVec');
    end
end

%% FIT MODEL ON SIMULATED DATA (NOT SUPERFREE MODEL)

clear all
nsimSubjs = 50; % how many simulated participants we are fitting

% load experimental settings
load('experimentalsettings.mat')

fitmodelVec = {'RL3WM_pers_0','Decay3_pers_0','RLWM_dn_pers_sub_0','RLWM_CA_pers_0','RLWM_dnwm_pers_sub_0','ns2_pers_0'};
simmodelVec = fitmodelVec;

nReps = 20;
for ifitmodel = 1:length(fitmodelVec)
    fitmodel = fitmodelVec{ifitmodel}
    
    [LB,UB,PLB,PUB,logflag,A,b,Aeq,beq,nonlcon,fixparams] = loadfittingparams(fitmodel);
    
    
    % get model title for simulation
    if strcmp(fitmodel(end-1:end),'_0') % if alpha- = 0
        modell = fitmodel(1:end-2);
        if strcmp(modell(end-3:end),'_sub') % if not fitting category condition
            modell = modell(1:end-4);
        end
    else
        modell = fitmodel;
    end
    
    switch fitmodel
        case 'RL3WMi_all_pers_0'
            modell = 'RL3WMi_pers';
            all_flag = 1;
        case 'RLWMi_dn_all_pers_sub_0'
            modell = 'RLWMi_dn_pers';
            all_flag = 1;
        otherwise
            all_flag = 0;
    end

    for isimmodel = 1:length(simmodelVec)
        simmodel = simmodelVec{isimmodel}
        
            for isubj = 1:nsimSubjs
                
                % load data
                load(sprintf('data/simdata/simdata_model_%s_subj%d.mat',simmodel,isubj))
                nParams = length(PLB);%+size(fixparams,2);
                
                x0_list = latinhs(nReps,nParams,PLB,PUB,[],1e3);
                xbestmat = nan(nReps,nParams);
                LLvec = nan(1,nReps);
                for irep = 1:nReps
                    x0 = x0_list(irep,:);
                    
                    eval(sprintf('[xbestmat(irep,:),LLvec(irep),~,~] = fmincon(@(x) -calc_LL_%s(x,stimvaluesCell,corrCell,responseCell,condVec,logflag,fixparams),x0,A,b,Aeq,beq,LB,UB);',modell));
                end
                
                try
                    tempx = xbestmat;
                    tempx(:,logflag) = exp(tempx(:,logflag));
                    templl = LLvec;
                    load(sprintf('fits/modelrecovery/fitmodel_%s_simmodel_%s_subj%d.mat',fitmodel,simmodel,isubj),'xbestmat','LLvec');
                    xbestmat = [xbestmat; tempx];
                    LLvec = [LLvec templl];
                catch
                    xbestmat(:,logflag) = exp(xbestmat(:,logflag));
                end
                
                if ~exist('fits/modelrecovery', 'dir')
                    mkdir('fits/modelrecovery')
                end
                save(sprintf('fits/modelrecovery/fitmodel_%s_simmodel_%s_subj%d.mat',fitmodel,simmodel,isubj),'xbestmat','LLvec');
            end
    end
end

%% GET BEST FIT PARAMS FOR EACH PARTICIPANT

clear all

nsimSubjs = 50;
load('experimentalsettings.mat')

% fitting subset of total models
% modelVec = {'RL3WM3_pers_0','superfree_dnRLWM_pers_0'};
modelVec = {'RL3WM_pers_0','Decay3_pers_0','RLWM_dn_pers_sub_0','RLWM_CA_pers_0','RLWM_dnwm_pers_sub_0','ns2_pers_0','RL3WM3_pers_0','superfree_dnRLWM_pers_0'};
nModels = length(modelVec);

for isimmodel = 1:nModels
    simmodel = modelVec{isimmodel};
    
    for ifitmodel = 1:nModels
        fitmodel = modelVec{ifitmodel};
        
        LLVec = nan(1,nsimSubjs);
        %  xbestMat = nan(nsimSubjs,nParamsVec(ifitmodel));

        for isubj = 1:nsimSubjs
            load(sprintf('fits/modelrecovery/fitmodel_%s_simmodel_%s_subj%d.mat',fitmodel,simmodel,isubj));
            
            idx = find(LLvec == min(LLvec),1,'first');
            LLVec(isubj) = LLvec(idx);
            xbestMat(isubj,:) = xbestmat(idx,:);
        end
        
        save(sprintf('fits/modelrecovery/fitmodel_%s_simmodel_%s.mat',fitmodel,simmodel),...
            'xbestMat','LLVec')
        
        clear xbestMat
        
    end
end