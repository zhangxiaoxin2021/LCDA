function G=updateG(W)

[D]=size(W,1);
G=eye(D);

for i=1:D
    a=2*norm(W(i,:));
    if a~=0
        G(i,i)=1/a;
    else
        G(i,i)=0;
    end
end

end