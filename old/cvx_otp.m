clc;clear all;close all;

% m = 20; n = 10; p = 4;
% A = randn(m,n); b = randn(m,1);
% C = randn(p,n); d = randn(p,1); e = rand;
% cvx_begin
%     variable x(n)
%     minimize( norm( A * x - b, 2 ) )
%     subject to
%         C * x == d
%         norm( x, Inf ) <= e
% cvx_end

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




% x_1=x_1-440;
% y_1=y_1-320;
% x_2=x_2-440;
% y_2=y_2-320;

% q=340:670;
% qq=220:470;
q=440:500;
qq=320:380;
I1=I1(q,qq);
I2=I2(q,qq);
Vx=Vx(q,qq);
Vy=Vy(q,qq);



dirak=Vx>0|Vy>0;

d=bwdist(dirak);
prah=10;
okolik=0.9;
d(d>prah)=prah;
d=prah-d;
d=d/(1/okolik*prah);
d=d+(1-okolik);
d=double(d);
% imshow(d,[])

% u(dirak) = Vx(dirak);
% v(dirak) = Vy(dirak);
[nonzero_x,nonzero_y]=find(dirak);

u0=Vx;
v0=Vy;

[fx, fy, ft] = computeDerivatives(I1, I2);


% Ix=Ix(:);
% Iy=Iy(:);
% It=It(:);


% n=length(Ix);
[m,n]=size(fx);

% cvx_begin
%     variable u(n)
%     variable v(n)
%     minimize( square(multiply(Ix,u)+multiply(Iy,v)+It)+
%     subject to
%         C * x == d
%         norm( x, Inf ) <= e
% cvx_end



% cvx_begin
% variable x(n)
% Objective = 0.5*sum_square(x-y)
% for i=1:n-2
%   Objective = Objective + norm([x(i)-x(i+1);x(i+1)-x(i+2)])
% end
% minimize(Objective)
% % insert any other constraints
% cvx_end

tv=linop_TV( [m,n], 'regular', 'cvx' );

cvx_begin
    variable u(m,n)
    variable v(m,n)
%     Objective = 1*(sum(tv(u))+sum(tv(v)));
%     Objective=sum(sum((fx.*u+fy.*v+ft).^2));
%     Objective=sum(sum(abs((fx.*u+fy.*v+ft))));
    Objective=0;
    
%     for k=1:m-1
%         disp(k)
%         for kk=1:n-1
%             Objective = Objective + norm([u(k,kk)-u(k+1,k);u(k,kk)-u(k,k+1)]);
%             Objective = Objective + norm([v(k,kk)-v(k+1,k);v(k,kk)-v(k,k+1)]);
%         end 
%     end

%     bjective = Objective + norm(u(2:n-1,:)-u(2:n,:)+)
%     Objective = Objective + norm(4*u(2:n-1,2:n-1))
    
%     Objective = Objective + sum(norm(2*u(1:n-1,1:n-1)-u(2:n,1:n-1)-u(1:n-1,2:n)));
%     Objective = Objective + sum(norm(2*v(1:n-1,1:n-1)-v(2:n,1:n-1)-v(1:n-1,2:n)));
%     p1=v(1:m-1,1:n-1)-v(2:m,1:n-1);
%     p2=v(1:m-1,1:n-1)-v(1:m-1,2:n);
    p1=(u(1:m-1,1:n-1)-u(2:m,1:n-1)).*d(1:m-1,1:n-1);
    p2=(u(1:m-1,1:n-1)-u(1:m-1,2:n)).*d(1:m-1,1:n-1);
    
    Objective = Objective + sum(sum(p1.^2+p2.^2));
%     Objective = Objective + sum(sum(abs(p1)+abs(p2)));
%     Objective = Objective + sum(sum(p1+p2));
    
%     p1=v(1:m-1,1:n-1)-v(2:m,1:n-1);
%     p2=v(1:m-1,1:n-1)-v(1:m-1,2:n);
    p1=(v(1:m-1,1:n-1)-v(2:m,1:n-1)).*d(1:m-1,1:n-1);
    p2=(v(1:m-1,1:n-1)-v(1:m-1,2:n)).*d(1:m-1,1:n-1);
    
    Objective = Objective + sum(sum(p1.^2+p2.^2));
%     Objective = Objective + sum(sum(abs(p1)+abs(p2)));
%     Objective = Objective + sum(sum(p1+p2));

    minimize(Objective)
    subject to
%         sum_square((u-u0).*dirak)==0
%         sum_square((v-v0).*dirak)==0
        for k=1:length(nonzero_y)
            u(nonzero_x(k),nonzero_y(k))-u0(nonzero_x(k),nonzero_y(k))==0;
            v(nonzero_x(k),nonzero_y(k))-v0(nonzero_x(k),nonzero_y(k))==0;
            
        end

cvx_end


hold off    
imshow(I1,[0,0.9])
hold on
quiver(u,v,'b','AutoScale','off')
quiver(u0,v0,'r','AutoScale','off')




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

