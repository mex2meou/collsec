% from python
% A = np.random.random_integers(2,size=(40,30))
% d,l np.where((A-1)==1)
% A = np.array(zip(d,l)) + 1
% k,l,Nx,Ny,Qx,Qy,Dnz = octave.cross_association(A)
function [k,l,Nx,Ny,Qx,Qy,Dnz] = cross_association(A)
A(:,3) = 1;
A = spconvert(A);
[k,l,Nx,Ny,Qx,Qy,Dnz] = cc_search(A,'hellscream',false)