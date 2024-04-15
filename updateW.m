function W=updateW(STLC,DTLc,L_sim,L_dgs,Gr,gamma,dim)

l=[eye(size(STLC,1)/2);-eye(size(STLC,1)/2)];
L=l*l'; 
I=eye(size(STLC,1));
Q=STLC+L+L_sim+gamma*Gr;
T=DTLc+L_dgs+1e-5*I;    %aviod singular 
[C,~] = eigs(Q,T,dim,'sm');
W=real(C);

end