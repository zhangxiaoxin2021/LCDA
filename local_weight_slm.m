function [Sw,Sb]=local_weight_slm(X,Y,k)
[n,m]=size(X);
Sb=zeros(m,m);Sw=zeros(m,m);
for i=1:n
    Xs=X(Y==Y(i),:); 
    Xd=X(Y~=Y(i),:);
    [es] =KNN(X(i,:),Xs,k);
    [ed] =KNN(X(i,:),Xd,k);
    xs=Xs(es,:);
    xd=Xd(ed,:);
    mxs=mean(xs);
    mxd=mean(xd);      
    Sb=Sb+((X(i,:)-mxd)'*(X(i,:)-mxd));
    Sw=Sw+((X(i,:)-mxs)'*(X(i,:)-mxs));
end
end



function [e] =KNN(in,test,k)
% in:       training samples data
% test:     testing data
% target:   k-nearist neighbors given by knn
% k:        the number of neighbors
    dist=zeros(size(test,1),1);
for i=1:size(test,1)
     dist(i)=norm(in-test(i,:),2);
end   
    [sdist,index]=sort(dist);
     e=index(2:k+1);
end    