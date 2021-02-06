%%%% INTRODUCCIÓN %%%%

%% Definir y representar señales

% x[n] = 2n if -3 <= n <= 3 else 0
n1=[-3:3];
x1=2*n1;
%stem(n1, x1);

% Extendemos el rango de visualización
n2=[-5:5];
x2=[0 0 x1 0 0];
%stem(n2, x2);

% Extendemos más aún el rango
n3=[-100:100];
x3=[zeros(1, 95) x2 zeros(1,95)];
%stem(n3,x3);

% Representamos las tres señales en tres figuras
figure('Name', 'n[-3:3]')
stem(n1, x1)
figure('Name', 'n[-5:5]')
stem(n2, x2)
figure('Name', 'n[-100:100]')
stem(n3, x3)

%% Representación de varias señales en un rango

% Una delta y una delta desplazada
nx1=[0:10];
x1=[1 zeros(1,10)];
nx2=[-5:5];
x2=[zeros(1,3) 1 zeros(1,7)];
figure; stem(nx1, x1);
figure;stem(nx2,x2);

% Sin decirle los índices, asume que n=[1,2,...,len(x)] !!
figure;stem(x1);
figure;stem(x2);

%% Señales en tiempo continuo

% Representación muestreando cada 0.1
t=[-5:0.1:5];
%t=linspace(-5,5,101)
x=sin(pi*t/4);
plot(t,x);

% Representación conjunta v1
t=[-4:1/8:4]'  %% vector columna
x1=sin(pi*t/4)
x2=cos(pi*t/4)
figure;plot(t, [x1 x2]);
figure;stem(t, [x1 x2]);

% Representación conjunta v2
t=[-4:1/8:4]  %% vector fila
figure;
plot(t, x1)
hold on
plot(t, x2)
hold off
figure;
stem(t, x1)
hold on
stem(t, x2)
hold off

%% Señales complejas

n=[0:32];
x=exp(j*(pi/8)*n);
% por defecto stem(n, x) pinta la parte real

% Pintamos parte real e imaginaria
figure;
stem(n, real(x))
hold on
stem(n, imag(x))
hold off

% Pintamos módulo y ángulo
figure;
stem(n, abs(x))
hold on
stem(n, angle(x))
hold off

%% Operaciones con señales

% Operaciones con dos señales que comparten los mismos índices
x1=sin((pi/4)*[0:30]);
x2=cos((pi/7)*[0:30]);

% Hay que poner un . para indicar que es elemento a elemento
y1=x1+x2;
y2=x1-x2;
y3=x1.*x2;
y4=x1./x2;
y5=2*x1;
y6=x1.^x2;

n=[0:30]
figure;stem(n, y1)
figure;stem(n, y2)
figure;stem(n, y3)
figure;stem(n, y4)
figure;stem(n, y5)
figure;stem(n, y6) % exponenciación compleja y solo muestra parte real (?)