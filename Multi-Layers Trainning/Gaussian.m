
function Z=Gaussian(u,v,x,y)
% u,vector,expactation;v,covariance matrix
%x=150:0.5:190;   
%y=35:110;       
[X,Y]=meshgrid(1:x,1:y);  
dx=std2(X); %X的方差
DX = sqrt(dx); 
dy=std2(std2); %Y的方差
DY = sqrt(dy);
COV=v(1,2);     %X Y的协方差
r=COV/(dx*dy);
part1=1/(2*pi*dx*dy*sqrt(1-r^2));
p1=-1/(2*(1-r^2));
px=(X-u(1)).^2./DX;
py=(Y-u(2)).^2./DY;
pxy=2*r.*(X-u(1)).*(Y-u(2))./(dx*dy);
Z=part1*exp(p1*(px-pxy+py));
mesh(x,y,Z);