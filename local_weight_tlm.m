function [Ts,Td]=local_weight_tlm(X,k)
[n,m]=size(X);
Sw=eye(m,m);
Sd=(eye(n)-((ones(n,1)*ones(n,1)')/n))*(eye(n)-((ones(n,1)*ones(n,1)')/n))';
Td=X'*Sd*X; 
for i=1:n
    [e] =KNN(X(i,:),X,k);
    xs=X(e,:);
    mxs=mean(xs);
    Sw=Sw+((X(i,:)-mxs)'*(X(i,:)-mxs));
end
Ts=Sw;
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