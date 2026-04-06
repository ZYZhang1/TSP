classdef TSP < ALGORITHM
% <multi/many> <real> <expensive><constrained/none>
% Kriging-Assisted CMOEA based on Two-Stage Probabilistic Penalty
% wmax  --- 20 --- Number of generations before updating Kriging models
% mu    ---  5 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya, A surrogate-
% assisted reference vector guided evolutionary algorithm for
% computationally expensive many-objective optimization, IEEE Transactions
% on Evolutionary Computation, 2018, 22(1): 129-142.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
%                                        
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            global phase v K Model_obj
            [wmax,mu] = Algorithm.ParameterSet(20,5);
            v         = 0;
            K         = 0;

            %% Generate the reference points and population
            NI             = Problem.N;
            P              = UniformPoint(NI,Problem.D,'Latin');
            A2             = SOLUTION(repmat(Problem.upper-Problem.lower,NI,1).*P+repmat(Problem.lower,NI,1));
            A1             = A2; 
            THETA_obj      = 5.*ones(Problem.M,Problem.D);
            THETA_con      = 5.*ones(size(A2.cons,2),Problem.D);
            Model_obj      = cell(1,Problem.M);
            Model_con      = cell(1,size(A2.cons,2));
            sample_success = 1;
            phase          = 1;
            Mean_CV_old    = inf; Mean_CV_new  = inf; Phase1_flag = 0; % phase 1 —>phase 2
            ND_Angle_old   = 0  ; ND_Angle_new = 0  ; Phase2_flag = 0; % phase 2 —>phase 3

            %% Optimization
            while Algorithm.NotTerminated(A2)
                % Refresh surrogate models
                if sample_success
                    [Model_obj,Model_con,THETA_obj,THETA_con] = model_train(A2,THETA_obj,THETA_con);
                end

                %Optimization
                [PopDec,PopObj,PopCon,ObjMSE,ConMSE] = optimizaiton(A1,wmax,Model_obj,Model_con);
                 
                % Select mu solutions for re-evaluation
                PopNew = NewSelect(PopDec,PopObj,PopCon,ObjMSE,ConMSE,A2,mu,Problem);

                sample_success = 0;
                if isempty(PopNew) == 0
                    A2             = [A2,PopNew];
                    sample_success = 1;
                    
                    % Phase Transistion
                    if sample_success
                        if phase == 1
                            if Problem.FE >=  NI + (Problem.maxFE - NI)/2
                                phase = 2;
                            elseif Problem.FE >=  NI + (Problem.maxFE - NI)/4
                                Mean_CV_new = sum(sum(max(0,PopNew.cons),2))/length(PopNew);
                                if Mean_CV_new > Mean_CV_old && isempty(find(all(PopNew.cons<=0,2), 1))
                                    if Phase1_flag == 0
                                        Phase1_flag = 1;
                                    elseif Phase1_flag == 1
                                        phase = 2;
                                    end
                                else
                                    Phase1_flag = 0;
                                end
                                Mean_CV_old = Mean_CV_new;
                            end
                        elseif phase == 2
                            if length(find(all(A2.cons<=0,2))) >= NI
                                phase = 3;
                            end
                        end
                        if phase == 3
                            %% Learning Distribution
                            PopDec = A2(all(A2.cons<=0,2)).decs;
                            v      = mean(PopDec,1);
                            K      = (PopDec-v)'*(PopDec-v)/(size(PopDec,1)-1);
                        end
                    end
                end

                % Population Update
                if sample_success
                    index = EnvironmentalSelection(A2.objs,A2.cons,NI);
                    A1    = A2(index);
                end
            end
        end
    end
end

function [Model_obj,Model_con,THETA_obj,THETA_con] = model_train(A2,THETA_obj,THETA_con)
    Dec = A2.decs;
    Obj = A2.objs;
    Con = A2.cons;
    Len_dec = size(Dec,2);
    Len_obj = size(Obj,2);
    Len_con = size(Con,2);
    for i = 1 : Len_obj
        [~,distinct1] = unique(round(Dec*1e10)/1e10,'rows');
        [~,distinct2] = unique(round(Obj(:,i)*1e10)/1e10,'rows');
        distinct = intersect(distinct1,distinct2);
        
        dmodel     = dacefit(Dec(distinct,:),Obj(distinct,i),'regpoly1','corrgauss',THETA_obj(i,:),1e-5.*ones(1,Len_dec),100.*ones(1,Len_dec));
        Model_obj{i}   = dmodel;
        THETA_obj(i,:) = dmodel.theta;
    end
    for i = 1 : Len_con
        [~,distinct1] = unique(round(Dec*1e10)/1e10,'rows');
        [~,distinct2] = unique(round(Con(:,i)*1e10)/1e10,'rows');
        distinct = intersect(distinct1,distinct2);
        
        dmodel     = dacefit(Dec(distinct,:),Con(distinct,i),'regpoly1','corrgauss',THETA_con(i,:),1e-5.*ones(1,Len_dec),100.*ones(1,Len_dec));
        Model_con{i}   = dmodel;
        THETA_con(i,:) = dmodel.theta;
    end
end

function [OffObj,Off_ObjMSE,OffCon,Off_ConMSE] = model_predict(Model_obj,Model_con,OffDec)
    N          = size(OffDec,1);
    Len_obj    = length(Model_obj);
    Len_con    = length(Model_con);
    OffObj     = zeros(N,Len_obj);
    OffCon     = zeros(N,Len_con);
    Off_ObjMSE = zeros(N,Len_obj);
    Off_ConMSE = zeros(N,Len_con);
      
    for i = 1 : N
        for j = 1 : Len_obj
            [OffObj(i,j),~,Off_ObjMSE(i,j)] = predictor(OffDec(i,:),Model_obj{j});
        end
        for j = 1 : Len_con
            [OffCon(i,j),~,Off_ConMSE(i,j)] = predictor(OffDec(i,:),Model_con{j});
        end
    end
    OffObj     = (OffObj);
    OffCon     = (OffCon);
    Off_ObjMSE = abs(Off_ObjMSE);
    Off_ConMSE = abs(Off_ConMSE);
end

function [PopDec,PopObj,PopCon,ObjMSE,ConMSE] = optimizaiton(A1,wmax,Model_obj,Model_con)
    PopDec = A1.decs;
    w      = 1;
    while w <= wmax
        OffDec = OperatorGA(PopDec);
        PopDec = [PopDec;OffDec];
        [PopObj,ObjMSE,PopCon,ConMSE] = model_predict(Model_obj,Model_con,PopDec);
        
        index  = SEnvironmentalSelection(PopDec,PopObj,ObjMSE,PopCon,ConMSE,length(A1));
      
        PopDec = PopDec(index,:);
        PopObj = PopObj(index,:);
        PopCon = PopCon(index,:);
        ObjMSE = ObjMSE(index,:);
        ConMSE = ConMSE(index,:);
        w = w + 1;
    end
end

