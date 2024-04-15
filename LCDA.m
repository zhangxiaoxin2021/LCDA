function [Ws,Wt]=LCDA(Xs,Xt,Ys,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%input Xs: source domain
%input Xt: target domain
%input Ys: source domain label
%input options: parameters value. For example: options.alpha=1;
%output Ws: source domain subspace feature extraction matrix
%output Wt: target domain subspace feature extraction matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parameters
alpha=options.alpha; %parameter SD
beta=options.beta;   %parameter TD
gamma=options.gamma; %parameter norm
k=options.k;         %parameter number of neighbors
d=options.dim;       %parameter number of subspace features
T=options.T;         %parameter numberof iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = length(unique(Ys));      %
[Dims,ns]=size(Xs);          %
[Dimt,nt]=size(Xt);          %
e_s=ones(ns,1)/ns;
e_t=ones(nt,1)/nt; 
n=ns+nt;                     %
X=blkdiag(Xs,Xt);            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%locality preservation
[L_s_sim,L_s_dsim]=local_weight_slm(Xs',Ys,k); %locality Source domian
[L_t_sim,L_t_gstr]=local_weight_tlm(Xt',k);
L_sim=blkdiag(alpha*L_s_sim,beta*L_t_sim);
L_dgs=blkdiag(alpha*L_s_dsim,beta*L_t_gstr);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize
e=[e_s;-e_t];
L0=e*e'*C;
%%%%
knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
Y_tar_pseudo = knn_model.predict(Xt');
Gr=eye(Dims+Dimt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%iteration
IsConverge = 0;
t=1;
while (IsConverge == 0 && t<T+1)
    %%%%%%%%%%%%%%%%%%%%%%
    %distribution alignment
	Lc = 0;Lcc=0;
	if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
		for c = reshape(unique(Ys),1,C)
			e = zeros(n,1);
			e(Ys==c) = 1 / length(find(Ys==c));
			e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
			e(isinf(e)) = 0;
			Lc = Lc + e*e';
        end

        for c1 = reshape(unique(Ys),1,C)
            for c2 = reshape(unique(Ys),1,C)
                if c1~=c2
                    e = zeros(n,1);
			        e(Ys==c1) = 1 / length(find(Ys==c1));
			        e(ns+find(Y_tar_pseudo==c2)) = -1 / length(find(Y_tar_pseudo==c2));
			        e(isinf(e)) = 0;
			        Lcc = Lcc + e*e';
                end
            end
        end
	end
    %%%%%%%%%%%%%%%%%%%%%%%%%
    LC=L0+Lc;
    LC = LC / norm(LC,'fro');
    Lcc = Lcc / norm(Lcc,'fro');
    STLc=X*LC*X';    %compute sum(X*Lc*X')
    DTLc=X*Lcc*X';   %compute X*sum(Lcc)*X')
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %update W
    W=updateW(STLc,DTLc,L_sim,L_dgs,Gr,gamma,d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %update Gr
    Gr=updateG(W);
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%update Y_tar_pseudo
    M=W'*X;
    Gs=M(:,1:ns);
    Gt=M(:,ns+1:n);
    knn_model = fitcknn(Gs',Ys,'NumNeighbors',1);
    Y_tar_pseudo = knn_model.predict(Gt');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    t=t+1;    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ws=W(1:Dims,:);
Wt=W(Dims+1:Dims+Dimt,:);
end


