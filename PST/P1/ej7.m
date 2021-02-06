% ej7.m

%% Transformaciones de la variable independiente

% Definimos una señal
nx=[-3:11]
x=[zeros(1, 3) 2 0 1 -1 3 zeros(1,7)]

% La representamos
stem(nx, x)
xlabel('Tiempo(Discreto)')
ylabel('Valor')
title('Señal discreta X[n]')

%% Hacemos varias transformaciones
y1=x
ny1=nx+2  % desplazamiento 2 uds a la derecha

y2=x
ny2=nx-1  % desplazamiento 1 uds a la izquierda

y3=x
ny3=-nx  % reflexión
%ny3=fliplr(nx) - 8

y4=x
ny4=-nx + 1  % reflexión y desplazamiento 1 ud a la izquierda
%ny4=-(nx-1)

%% Mostramos los resultados

figure; stem(nx, x); title('x')
figure;

subplot(2,2,1);
stem(ny1, y1)
title('x retrasada 2')
xlabel('ny1')

subplot(2,2,2);
stem(ny2, y2)
title('x adelantada 1')
xlabel('ny2')

subplot(2,2,3);
stem(ny3, y3)
title('x invertida')
xlabel('ny3')

subplot(2,2,4);
stem(ny4, y4)
title('x invertida y luego adelantada 1')
xlabel('ny4')