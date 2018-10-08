clc;clear all;close all;


I1=double(imread('imageForTPS.tif',1))/255;
I2=double(imread('imageForTPS.tif',2))/255;

t1=readtable('Choices.csv');
t2=readtable('savedtracks.csv');

pouzit=t2{:,4}>=0;

w0=t2{:,4};
w0=(3-w0)/3;
w0(w0<0)=0;


y_1=t1{pouzit,4};
x_1=t1{pouzit,5};

y_2=t2{pouzit,2};
x_2=t2{pouzit,3};


vx=x_2-x_1;
vy=y_2-y_1;

Vx=zeros(size(I1));
Vy=zeros(size(I1));

for k=1:length(vx)
    Vx(x_1(k),y_1(k))=vx(k);
    Vy(x_1(k),y_1(k))=vy(k);
end

I1=I1(340:670,220:470);
I2=I2(340:670,220:470);
Vx=Vx(340:670,220:470);
Vy=Vy(340:670,220:470);

ite=100000;

dirak=Vx>0|Vy>0;

angle=2*pi*rand(size(dirak));

u=mean(sqrt(vx.^2+vx.^2))*sin(angle);
v=mean(sqrt(vx.^2+vx.^2))*cos(angle);

u(dirak) = Vx(dirak);
v(dirak) = Vy(dirak);

% Estimate spatiotemporal derivatives
[fx, fy, ft] = computeDerivatives(I1, I2);

u0=Vx;
v0=Vy;



uv=zeros(size(u));
vv=zeros(size(u));

u([1 end],:)=0;
u(:,[1 end])=0;

v([1 end],:)=0;
v(:,[1 end])=0;


% Averaging kernel
kernel_1=[1/12 1/6 1/12;1/6 0 1/6;1/12 1/6 1/12];

% Iterations
for i=1:ite
    
   delta_u=del2(u);
   delta_v=del2(v);
%     
%     uAvg=conv2(u,kernel_1,'same');
%     vAvg=conv2(v,kernel_1,'same');
%     
%    
%     u= + uAvg - dirak.*(-u-u0)/alpha^2;
%     v= + vAvg - dirak.*(-v-v0)/alpha^2;     
    step=0.5;
    mom=0.9;
    alpha=1;
    beta=1;
    gama=10;
    
    dE=(gama*dirak.*(u-u0)-alpha*delta_u+ beta*fx .*( fx .* u + fy.*v + ft ));
    dE([1 end],:)=0;
    dE(:,[1 end])=0;
    
    
    uv=mom*uv+(1-mom)*dE;
    
    u= u - step*uv;
    
    dE=step*(+dirak.*(v-v0)-alpha*delta_v+ beta*fy .*( fx .* u + fy.*v + ft ));
    dE([1 end],:)=0;
    dE(:,[1 end])=0;
    
    
    vv=mom*vv+(1-mom)*dE;
    
    v= v - step*vv;  
% 
    

if mod(i,100)==0
hold off    
imshow(I1,[0,0.9])
hold on
title(i)
quiver(u,v,'b','AutoScale','off')
quiver(u0,v0,'r','AutoScale','off')
drawnow;
% pause(0.1)
%   
end
    
    
end



function [fx, fy, ft] = computeDerivatives(im1, im2)

if size(im2,1)==0
    im2=zeros(size(im1));
end

% Horn-Schunck original method
fx = conv2(im1,0.25* [-1 1; -1 1],'same') + conv2(im2, 0.25*[-1 1; -1 1],'same');
fy = conv2(im1, 0.25*[-1 -1; 1 1], 'same') + conv2(im2, 0.25*[-1 -1; 1 1], 'same');
ft = conv2(im1, 0.25*ones(2),'same') + conv2(im2, -0.25*ones(2),'same');

% derivatives as in Barron
% fx= conv2(im1,(1/12)*[-1 8 0 -8 1],'same');
% fy= conv2(im1,(1/12)*[-1 8 0 -8 1]','same');
% ft = conv2(im1, 0.25*ones(2),'same') + conv2(im2, -0.25*ones(2),'same');
% fx=-fx;fy=-fy;

% An alternative way to compute the spatiotemporal derivatives is to use simple finite difference masks.
% fx = conv2(im1,[1 -1]);
% fy = conv2(im1,[1; -1]);
% ft= im2-im1;

end

