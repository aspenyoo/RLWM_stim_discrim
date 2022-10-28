 function [c,ceq] = interferencecon(x)
% fmincon has to satisfy c(x) ≤ 0 and ceq(x) = 0

c(1) = x(6) *  x(9) - 1;
c(2) = x(6) * x(10) - 1;
c(3) = x(6) * x(11) - 1;
c(4) = x(7) *  x(9) - 1;
c(5) = x(7) * x(10) - 1;
c(6) = x(7) * x(11) - 1;

ceq = [];

