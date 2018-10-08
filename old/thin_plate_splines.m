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



u0=Vx;
v0=Vy;

dirak=Vx>0|Vy>0;



% A=zeros(length(x_1));
% for k=1:length(x_1)
%     for kk=1:length(x_1)
%         
%         d=sqrt((x_1(k)-x_2(k)).^2+(y_1(k)-y_2(k)).^2);
%         if k~=kk
%             A(k,kk)=d*log(d+1e-6);
%         end
%         
%     end
% end

d=pdist2([x_1,y_1],[x_2,y_2])';
A=d.*log(d+1e-6);


V = [ones(length(x_1), 1), x_1,y_1]';

% Target points.
y = [x_2,y_2] ;

M = [[A, V']; [V, zeros(2+1, 2+1)]];
Y = [y;zeros(2+1, 2)];

X = M\Y;

mapping_coeffs = X(1:end-(2+1),:);
poly_coeffs = X((end-2):end,:);


[Y,X]=meshgrid(1:size(I1,2),1:size(I1,1));

d=pdist2([x_1,y_1],[X(:),Y(:)])';
A=d.*log(d+1e-6);
V = [ones(size(X(:),1), 1), [X(:),Y(:)]];
f_surface = [A V] * [mapping_coeffs; poly_coeffs];

fX = reshape(f_surface(:, 1), size(X));
fY = reshape(f_surface(:, 2), size(X));


hold off    
imshow(I1,[0,0.9])
hold on
quiver(fX-X,fY-Y,'b','AutoScale','off')
quiver(u0,v0,'r','AutoScale','off')



u=fX-X;
v=fY-Y;
[m,n]=size(u);