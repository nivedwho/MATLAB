n = "Enter the number of Sides:";
N = input(n);

A= zeros(1,N);
x=A;
y=A;
for s=1:N
   B ="Enter x coordinate:";
   C ="Enter y coordinate";
   b= input(B);
   c= input(C);
   x(1,s)= b;
   y(1,s)= c;
end
    
% check if inputs are same size
if ~isequal( size(x), size(y) )
  error( 'X and Y must be the same size');
end
 
% temporarily shift data to mean of vertices for improved accuracy
xm = mean(x);
ym = mean(y);
x = x - xm;
y = y - ym;
  
% summations for CCW boundary
xp = x( [2:end 1] );
yp = y( [2:end 1] );
a = x.*yp - xp.*y;
 
A = sum( a ) /2;
xc = sum( (x+xp).*a  ) /(6*A);
yc = sum( (y+yp).*a  ) /(   6*A);
Ixx = sum( (y.*y +y.*yp + yp.*yp).* (x.*yp - xp.*y)  /12)*(-1);
Iyy = sum( (x.*x +x.*xp + xp.*xp).* (x.*yp - xp.*y)  /12)*(-1);
 
% centroidal moments
Iuu = Ixx + A*yc*yc;
Ivv = Iyy + A*xc*xc;
 
% replace mean of vertices
x_cen = xc + xm;
y_cen = yc + ym;
Ixx = (Iuu + A*y_cen*y_cen)*(-1);
Iyy = (Ivv + A*x_cen*x_cen)*(-1) ;
 

% return values
disp("Centroid point:");
Centroid = [x_cen  y_cen]
disp("Moment of Inertia on X axis and  Y axis:");
I = [Ixx Iyy]
disp("Moment of inertia on Centroidal Axis:");
Ic = [Iuu Ivv]