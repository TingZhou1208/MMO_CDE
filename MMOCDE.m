function [ps,pf]=MMOCDE(func_name,VRmin,VRmax,n_obj,NP,Max_Gen )
%% 初始化参数
n_var=size(VRmin,2);    %Obtain the dimensions of decision space
Max_FES=Max_Gen*NP;     %Maximum fitness evaluations
%  F=0.9;%变异率,缩放因子（0,1.2]
CR=0.7;%交叉参数
count(200,3)=0;
AA=[];
BB=[];
%% Initialize population
VRmin=repmat(VRmin,NP,1);
VRmax=repmat(VRmax,NP,1);
pos=VRmin+(VRmax-VRmin).*rand(NP,n_var); %initialize the positions of the individuals
%% Evaluate the population
fitness=zeros(NP,n_obj);
for i=1:NP
    fitness(i,:)=feval(func_name,pos(i,:));
end
fitcount=NP;            % count the number of fitness evaluations
pop=[pos,fitness];
%% 初始化外部存档
pop1=non_domination_scd_sort(pop,n_obj,n_var);
EXA= pop1(pop1(:,n_var+n_obj+1)==1,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%循环
for I=1:Max_Gen
    fitpop= non_domination_scd_sort(pop,n_obj,n_var);
    for a=1:NP
        for b=1:NP
            if pos(a,1:n_var)== fitpop(b,1:n_var)
                p(a)=fitpop(b,n_var+n_obj+1);
                fitpop(b,1:n_var)=inf;
                break;
            end
        end
    end
    %% 构造小生境
    nsize_k=5;%%领域半径
    nsize=2*nsize_k+1;%%领域大小
    nig=cell(NP,1);
    for i=1:NP
        %% 左邻居
        if i-nsize_k<1
            nig{i,1}=[nig{i,1} 1:i-1 NP+i-nsize_k:NP];
        else
            nig{i,1}=[nig{i,1} i-nsize_k:i-1];
        end
        %% 右邻居
        if i+nsize_k>NP
            nig{i,1}=[nig{i,1} i+1:NP 1:nsize_k+i-NP];
        else
            nig{i,1}=[nig{i,1} i+1:i+nsize_k];
        end
        %% 本身
        nig{i,1}=[nig{i,1} i];
    end
    nig=cell2mat(nig);
    %% 变异
    for i=1:NP
        middlepop=zeros(nsize,n_var+n_obj);  %%最后一个个体是本身
        for ii=1:nsize             %%把前四个个体形成自己的邻居
            middlepop(ii,1:n_var+n_obj)=pop(nig(i,ii),1:n_var+n_obj);
        end
        
        sortmiddlepop=non_domination_scd_sort(middlepop,n_obj,n_var);
        nbest=sortmiddlepop(1,1:n_var+n_obj); %%找到领域最优
        
        rep=middlepop(1:nsize-1,1:n_var+n_obj); %%返回Y中有二selected_pop没有的行响亮一第一列升序排序返回
        dx=randperm(nsize-1);
        %% 计算在决策空间中当前个体与其他个体的欧式距离
        newpop=rep;             %the population to calculate distance to the current particle
        poptemp=repmat(pop(i,1:n_var),size(newpop,1),1); %%取出每个个体的的函数值
        distance_var=sqrt(sum((newpop(:,1:n_var)-poptemp(:,1:n_var)).*(newpop(:,1:n_var)-poptemp(:,1:n_var)),2));%%根据目标函数值来计算当前个体与种群之间的欧式距离（目标空间）
        [bb,cc]=sort(distance_var);       %%从小到大排序，输出排序后所对应的个体的索引
        nearest=newpop(cc(1),1:n_var+n_obj);    %%找到领域中距离自己最近的个体
        
        repp=newpop(cc(2:end),1:n_var+n_obj);
        nsize1=size(repp,1);
        dx1=randperm(nsize1);
        %         %% 测试1电脑34
        F1= p(i)./max(p);
        F=normrnd(F1,0.1);
        
        if p(i)==1
            %DE/best/2   还可以(已经测试)（进一步测试)
            count(I,1)=count(I,1)+1;
            Meta(i,:)= nbest(1,1:n_var)+F*(rep(dx(1),1:n_var)-rep(dx(2),1:n_var)+rep(dx(3),1:n_var)-rep(dx(4),1:n_var));   %生成变异矢量Xm
        else
            count(I,2)=count(I,2)+1;
            %DE/rand-best/1 不好(已经测试)
            Meta(i,:)= nearest(1,1:n_var)+F*(repp(dx1(1),1:n_var)-repp(dx1(2),1:n_var)+repp(dx1(3),1:n_var)-repp(dx1(4),1:n_var));   %生
        end
        %%边界处理
        Meta(i,:) = boundConstraint(Meta(i,:), pos(i,:), [VRmin(1,:);VRmax(1,:)]);
    end
    trial=pos;
    %% 交叉操作
    for i=1:NP
        r=randperm(n_var,1);
        for j=1:n_var
            if rand<=CR||j==r
                trial(i,j)=Meta(i,j);
            end
        end
    end
    %% 评估种群
    for i=1:NP
        trialfitness(i,:)=feval(func_name,trial(i,:));
        fitcount=fitcount+1;
    end
    trialpop=[trial,trialfitness];
    %% 非支配解排序
    tempop=[trialpop;pop];
    %% 环境选择 对重组解进行聚类
    [custer,num]=AP(tempop,n_obj, n_var);
    %% 在每个子种群中选择
    GBA=[];
    subgbest=[];
    tem_dele=[];
    for ij=1:num
        temp_pop=non_domination_scd_sort(custer{ij,1}, n_obj, n_var);
        tempindex=find(temp_pop(:,n_var+n_obj+1)==1);
        GBA{ij,1}=temp_pop(tempindex,1:n_var+n_obj);
        subgbest{ij,1}=temp_pop(1,1:n_var+n_obj);
        temp_pop(tempindex,:)=[];
        tem_dele{ij,1}=temp_pop(:,1:n_var+n_obj);
    end
    
    tempGBA=cell2mat(GBA);
    tempdele=cell2mat(tem_dele);
   % N_size=size(tempGBA,1);
    %第二判定标准选x空间距离，按x空间拥挤距离降序排列，选前几个
    [ C,IA,IC]= unique(tempGBA(:,1:n_var),'rows');%%v decision
   % N_size=length(IA);
     tempGBA= tempGBA(IA,:);
     N_size=size(tempGBA,1);

    if N_size>NP
        newpop1=non_domination_scd_sort(tempGBA,n_obj,n_var);        
%         %pop =newpop1(1:NP,1:n_var+n_obj);
%         AA{I}=newpop1
        newpop1=replace_decision_chromosome_kmeans(newpop1,n_obj,n_var,NP);
        pop =newpop1(:,1:n_var+n_obj);
%         BB{I}=pop
    elseif N_size<NP
        remaining=NP-N_size;
        CrowdDis=Crowding(tempdele(:,1:n_var));
        tempdele(:,n_var+n_obj+1)=CrowdDis;
        [aa,dd]=sort(tempdele(:,n_var+n_obj+1),'descend');
        newpop2=tempdele(dd(1:remaining),:);
        pop=[tempGBA;newpop2(:,1:n_var+n_obj)];
    else
        pop=tempGBA;
    end
    pos=pop(:,1:n_var);
    %% UPdate EXA
    tempEXA=[pop(:,1:n_var+n_obj);EXA(:,1:n_var+n_obj)];
    tempEXA=unique(tempEXA,'rows','stable');%%过滤掉相同的个体
    tempEXA=non_domination_scd_sort(tempEXA,n_obj,n_var);
    tempEXA1=tempEXA(tempEXA(:,n_var+n_obj+1)==1,:);
    
    if size(tempEXA1,1)>NP
        EXA= replace_decision_chromosome_kmeans(tempEXA1,n_obj,n_var, NP);
    else
        EXA=tempEXA1;
    end
    
    
    
    %     I
    %     clf;
    %     figure(1)
    %     plot(pop(:,1),pop(:,2),'r+')
    %     pause(0.01)
    %         hold on
    %         figure(2)
    %         plot(pop(:,1),pop(:,2),'r+')
    %         pause(0.01)
    
    
    if fitcount>Max_FES
        break;
    end
end
%% Output
% ps = pop(pop(:,n_var+n_obj+1)==1,1:n_var);
% pf = pop(pop(:,n_var+n_obj+1)==1,n_var+1:n_var+n_obj);
% ps = pop(:,1:n_var);
% pf = pop(:,1+n_var:n_var+n_obj);
ps = EXA(EXA(:,n_var+n_obj+1)==1,1:n_var);
pf = EXA(EXA(:,n_var+n_obj+1)==1,n_var+1:n_var+n_obj);
end