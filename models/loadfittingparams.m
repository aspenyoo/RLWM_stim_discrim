function [LB,UB,PLB,PUB,logflag,A,b,Aeq,beq,nonlcon,fixparams] = loadfittingparams(model)


if strcmp(model(end-1:end),'_0')
    model = model(1:end-2);
    negalpha0_flag = 1;
else
    negalpha0_flag = 0;
end

try
    if strcmp(model(end-3:end),'_sub')
        model = model(1:end-4);
        dnsub_flag = 1;
    else
        dnsub_flag = 0;
    end
catch
    dnsub_flag = 0;
end

% try
%     if strcmp(model(1:9),'superfree')
%         super_flag = 1;
%         model = model(11:end);
%     else
%         super_flag = 0;
%     end
% catch
%     super_flag = 0;
% end


try
    if strcmp(model(end-3:end),'pers')
        pers_flag = 1;
        model = model(1:end-4);
        try
            if strcmp(model(end-3:end),'full')
                fullpers_flag = 1;
                model = model(1:end-5);
            else
                fullpers_flag = 0;
                model = model(1:end-1);
            end
        catch
            fullpers_flag = 0;
            model = model(1:end-1);
        end
    else
        pers_flag = 0;
    end
catch
    pers_flag = 0;
end

try
    if strcmp(model(end-3:end),'_all')
        model = model(1:end-4);
        all_flag = 1;
    else
        all_flag = 0;
    end
catch
    all_flag = 0;
end

try
    if strcmp(model(end-1:end),'_i')
        model = model(1:end-2);
        i_flag = 1;
    else
        i_flag = 0;
    end
catch i_flag = 0;
end

% get bounds and starting value fixparams = [];
nonlcon = []; fixparams = [];
switch model
    case 'WM'
        n = 2;
        A =[];
        b =[];
        Aeq =[];
        beq =[];
        logflag = logical([0 0]);
    case 'RL'
        n = 3;
        A =[];
        b =[];
        Aeq =[];
        beq =[];
        logflag = logical([1 1 0]);
    case 'RLWM'
        n = 6;
        A = [0 0 0 0 -1 1];
        b = 0;
        Aeq = [];
        beq = [];
        logflag = logical([1 1 0 0 0 0]);
    case {'dnRLWM'}
        n=7;
        A = [0 0 0 0 0 -1 1];
        b = 0;
        Aeq = [];
        beq = [];
        logflag = logical([1 1 0 0 0 0 0]);
        if (negalpha0_flag)
            fixparams = [2; 0];
        end
    case {'RL3WM','RL3WMi'}
        n=8;
        A = [0 0 0 0 0 0 -1 1];
        b = 0;
        Aeq = [];
        beq = [];
        logflag = logical([1 1 1 1 0 0 0 0]);
        if (negalpha0_flag)
            fixparams = [4; 0];
        end
    case {'RLWM_CA','Decay3'}
        n=8;
        A = [0 0 0 0 0 0 -1 1 ];
        b = 0;
        Aeq = [];
        beq = [];
        logflag = logical([1 1 0 0 0 0 0 0]);
        if (negalpha0_flag)
            fixparams = [2; 0];
        end
    case {'RLWM_dn','RLWM_dnwm','RLWMi_dn'}
        n=9;
        A = [0 0 0 0 0 0 0 -1 1 ];
        b = 0;
        Aeq = [];
        beq = [];
        logflag = logical([1 1 0 0 0 0 0 0 0]);
        if (negalpha0_flag)
            fixparams = [2; 0];
        end
%     case 'RL3WM_dnwm'
%         n=10;
%         A = [0 0 0 0 0 0 0 0 -1 1 ];
%         b = 0;
%         Aeq = [];
%         beq = [];
%         logflag = logical([1 1 1 1 0 0 0 0 0 0]);
    case 'RL3WM3'%,'RL3WM3i'}
        n=10;
        A = [0 0 0 0 0 0 0 0 -1 1];
        b = 0;
        Aeq = [];
        beq = [];
        logflag = logical([1 1 1 1 0 0 0 0 0 0]);
        if (negalpha0_flag)
            fixparams = [4; 0];
        end
    case 'ns2'
        n=8;
        A = [0 0 0 0 -1 1 0 0 ;...
             0 0 0 0 -1 0 1 0 ;...
             0 0 0 0 -1 0 0 1 ];
        b = [0; 0; 0];
        Aeq = [];
        beq = [];
        logflag = logical([1 1 0 0 0 0 0 0]);
        if (negalpha0_flag)
            fixparams = [2; 0];
        end
%     case {'RLWM_dndecay'}
%         n = 10;
%         A = [0 0 0 0 0 0 0 0 -1 1 ];
%         b = 0;
%         Aeq = [];
%         beq = [];
%         logflag = logical([1 1 0 0 0 0 0 0 0 0]);
end

if dnsub_flag
    switch model
        case {'RLWM_dn','RLWMi_dn','RLWM_dnwm'}
            fixparams = [fixparams [6; 0]];
        case 'Decay3'
            fixparams = [fixparams [5; 0]];
        case 'RL3WM3'
            fixparams = [fixparams [7; 0]];
    end
end

if pers_flag % if adding perseveration parameter(s)
    A = [A zeros(size(A,1),2)];
    logflag = [logflag false false];
    n=n+2;
    if ~fullpers_flag
    fixparams = [fixparams [n; 1]]; % if fixing tau to 1
    end
end

LB = zeros(1,n)+eps; 
UB = ones(1,n)-eps; 
PLB = 0.1*ones(1,n); 
PUB = 0.9*ones(1,n); 

if pers_flag
    LB(end-1) = -1;
end

% RLWMii models (condition-specific interference weight)
if i_flag
    A = [A zeros(size(A,1),3)];
    logflag = [logflag false false false];
    LB = [LB 1e-3*ones(1,3)]; 
    UB = [UB 5 5 5];
    PLB = [PLB 0.5 0.5 0.5];
    PUB = [PUB 2 2 2];
    n = n+3;
    nonlcon = @(x) interferencecon(x);
end

% beta for beta_test, when learn and test phase fit together
if all_flag 
    A = [A zeros(size(A,1),1)];
    logflag = [logflag false];
    n = n+1;
    LB = [LB 1e-3];
    UB = [UB 150];
    PLB = [PLB 1e-3];
    PUB = [PUB 10];
end

% beta_RL
A = [A zeros(size(A,1),1)];
LB = [LB 1e-3]; 
UB = [UB 150]; 
PLB = [PLB 1e-3]; 
PUB = [PUB 10]; 
logflag = [logflag false];
fixparams = [fixparams [n+1; 100]];

% logflag and fixparams stuff
LB(logflag) = log(LB(logflag));
UB(logflag) = log(UB(logflag));
PLB(logflag) = log(PLB(logflag));
PUB(logflag) = log(PUB(logflag));

if ~(isempty(fixparams))
    % vector of non-fixed parameter indices
    freeparamsidx = 1:(n+1);
    freeparamsidx(fixparams(1,:)) = [];
    
    if ~isempty(A)
        if numel(A) == max(size(A))
            A(fixparams(1,:)) = [];
        else
            A(:,fixparams(1,:)) = [];
        end
    end
    logflag(fixparams(1,:)) = [];
    LB(fixparams(1,:)) = [];
    UB(fixparams(1,:)) = [];
    PLB(fixparams(1,:)) = [];
    PUB(fixparams(1,:)) = [];
end

% if super_flag
%         LB = [LB LB LB];
%         UB = [UB UB UB];
%         PLB = [PLB PLB PLB];
%         PUB = [PUB PUB PUB];
%         logflag = [logflag logflag logflag];
%         A = [A A A];
%         b,
%         Aeq,
%         beq
%         fixparams = [fixparams fixparams fixparams];
% end
