function Next = SEnvironmentalSelection_many(PopDec,PopObj,ObjMSE,PopCon,ConMSE,N)
     %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort_TSP(PopDec,PopObj,ObjMSE,PopCon,ConMSE,N);
    
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);

    zmin = min(PopObj);zmax = max(PopObj);
    PopObj = (PopObj - zmin)./max(zmax - zmin,10e-10);
    %% Select the solutions in the last front
    if MaxFNo == 1
        Del  = Truncation(PopObj(Last,:), N);
        Next(Last(Del)) = true; 
    else
        Choose = Dist_Selection(PopObj(Next,:), PopObj(Last,:), N-sum(Next));
        Next(Last(Choose)) = true;
    end
end

function Choose = Dist_Selection(PopObj1,PopObj2,mu)
    PopObj = [PopObj1;PopObj2];
    N      = size(PopObj,1);
    N1     = size(PopObj1,1);
    N2     = size(PopObj2,1);
    
    %% Calculate the shifted distance between each two solutions
    Distance = acos(1-pdist2(PopObj,PopObj,'cosine'));
    Distance(logical(eye(length(Distance)))) = inf;
    
%     for i = 1 : N
%         for j = i+1 : N
%             %             Distance(i,j) = norm(PopObj(i,:)-PopObj(j,:),2);
%             Distance(i,j) = acos(1-pdist2(PopObj(i,:),PopObj(j,:),'cosine'));
%             Distance(j,i) = Distance(i,j);
%         end
%     end
    
    %% Calculate D
    Next1 = 1:N1;
    Next2 = N1+1:N;
    for i = 1 : mu
        Distance1 = sort(Distance(Next2,Next1),2);
        [~,index] = max(Distance1(:,1));
        Next1     = [Next1,Next2(index)];
        Next2(index) = [];
    end
    Choose = Next1(N1+1:end) - N1;
end


function Del = Truncation(PopObj,K)
    %% Select part of the solutions by truncation
    [N,~]  = size(PopObj);
    
    %% Calculate the shifted distance between each two solutions
    Distance = acos(1-pdist2(PopObj,PopObj,'cosine'));
    
%     Distance = inf(N);
%     for i = 1 : N
%          for j = i+1 : N
%             Distance(i,j) = acos(1-pdist2(PopObj(i,:),PopObj(j,:),'cosine'));
%             Distance(j,i) = Distance(i,j);
%         end
%     end
    %% Truncation
    Distance(logical(eye(length(Distance)))) = inf;
    Del = true(1,N);
    while sum(Del) > K
        Remain   = find(Del);
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = false;
    end
end