%% Gráficas

nx1 = [0:9]
nh1 = [0:4]
nh2 = [0:4]

x1 = [1 1 1 1 1 zeros(1, 5)]
h1 = [1 -1 3 0 0]
h2 = [0 2 5 4 -1]

stem(nx1, x1);figure;
stem(nh1, h1);figure;
stem(nh2, h2);

%% Propiedad conmutativa
nc = [0:13]
c1 = conv(x1, h1)
c2 = conv(h1, x1)

soniguales_c = isequal(c1, c2)

stem(nc, c1); figure;
stem(nc, c2);

%% Propiedad distributiva

nd = [0:13]
d1 = conv(x1, h1 + h2)
d2 = conv(x1, h1) + conv(x1, h2)

soniguales_d = isequal(d1, d2)

stem(nd, d1); figure;
stem(nd, d2);

%% Propiedad asociativa

% pintar w
nw = [0:13]
w = conv(x1, h1)
stem(nw, w); figure

% pintar y1
ny1 = [0:17]
y1 = conv(w, h2)
stem(ny1, y1);
title('y1');
figure;

% pintar hseries
nhseries = [0:8]
hseries = conv(h1, h2)
stem(nhseries, hseries);figure

% pintar y2
ny2 = [0:17]
y2 = conv(x1, hseries)
stem(ny2, y2);
title('y2')

%% Apartado e
nhe1 = nh1
he1 = h1
nhe2 = nhe1 - 2
he2 = he1

nye2 = [-2:11]
nye1 = [0:13]
ye2 = conv(x1, he2)
ye1 = conv(he1, x1)

stem(nye2, ye2); 
title('ye2')
figure;
stem(nye2, ye1) % ye1[n-2]
title('ye1[n-2]')

%% Apartado f
nw = nx1
w=(nw + 1).*x1
stem(nw, w);

nyf1 = [0:13]
yf1 = conv(w, h1)
figure;
stem(nyf1, yf1)
title('yf1')

nhf1=[0:4]
hf1=[1 0 0 0 0]
figure; stem(nhf1, hf1)

nhseries= [0:8]
hseries = conv(hf1, h1)
figure; stem(nhseries, hseries)

nyf2= [0:17]
yf2 = conv(x1, hseries)
figure; stem(nyf2, yf2)
title('yf2')
% No sale lo mismo, pero no hay contradicción, ya que el sistema 1 no es LTI.

%% Apartado g

nxg = [0:4]
xg = [2 0 0 0 0]
yga=xg.^2
nyga = nxg
figure; stem(nyga, yga)
title('yga')

nygb = [0:8]
ygb = conv(xg, h2)
figure; stem(nygb, ygb)
title('ygb')

nyg1 = nygb
yga = [yga zeros(1, 4)]
yg1 = yga + ygb
figure; stem(nyg1, yg1)
title('yg1')

nhg1 = [0:4]
hg1 = [2 0 0 0 0]
figure; stem(nhg1, hg1)
title('hgl')

nhpar = nh2
hpar = hg1 + h2
figure; stem(nhpar, hpar)
title('hparallel')

nyg2 = [0:8]
yg2 = conv(xg, hpar)
figure; stem(nyg2, yg2)
title('yg2')

xgh=conv(xg, hg1)
% Coinciden yg1 e yg2 por casualidad