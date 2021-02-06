% ej6.m
% Representar una señal discreta junto a su media

n = [0:16];
x1 = cos(pi*n/4);
y1 = mean(x1);
stem(n,x1,'r')
title('x1[n] = cos(pi*n/4) & media')
xlabel('Tiempo (Discreto)')
ylabel('x1[n]')
hold on
m1=y1*ones(1,17);
plot(n,m1,'g')
hold off
legend('Cos (pi*n/4)', 'Media (Cos (pi*n / 4))');

% Mostramos como utilizar una función
n=[0:15];
x1=4*sin((pi/4)*n);
[y1,z1]=f_ej6(x1);
figure;
stem(n,x1);
hold on;
stem(n,y1,'r');
stem(n,z1,'g');
hold off;