
%% Ejercicio 3
N = 128
nn = 0:40;
a = 0.88 * exp(j*2*pi/5);
xn = a.^nn;

plot_dtft(xn, N)

%% Ejercicio 4
N = 128
n = -10:10
x = [ones(1, 21)]

w0 = 2*pi/sqrt(31)
e = exp(j*w0*n)

plot_dtft(x, N)
plot_dtft(x.*e, N)

[H, W] = dtft(x.*e, N)
[argvalue, argmax] = max(abs(H));
-1 + (argmax-1)/64 
w0 / pi
%% Apartado b
w0 = 2*pi + pi/2
e = exp(j*w0*n)

plot_dtft(x, N)
plot_dtft(x.*e, N)

%% Apartado c
w0 = 2*pi/sqrt(31)


plot_dtft(x, N)
plot_dtft(x.*cos(w0*n), N)

% Doble banda porque se suman dos señales -- descomposicion del coseno
% El maximo es distinto porque es el máximo del modulo.
plot_dtft(x.*(0.5*exp(j*w0*n)),N)
plot_dtft(x.*(0.5*exp(-j*w0*n)),N)

%% Ejercicio 5
L = 64
N = 1024
n = 0:L-1
w = [ones(1, L)]
w0 = 2*pi/sqrt(L)
x = exp(j * w0 * n)

plot_dtft(x, N)
plot_dtft(w, N)
plot_dtft(x.*w, N)

%% Apartado b

N = 1024
w0 = 2*pi/sqrt(31)
w1 = [ones(1, 32)]
w2 = [ones(1, 64)]
w3 = [ones(1, 128)]
w4 = [ones(1, 256)]

[H, W] = dtft(w1, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,1)
plot(norm_freq, mod)
title('Magnitude Response')
ylabel('|H(w)|, L=32')

L = 0:31
x = exp(j * w0 * L)
[H, W] = dtft(x.*w1, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,2)
plot(norm_freq, mod)
title('Magnitude Response')
ylabel('|H(w)X(w)|')


[H, W] = dtft(w2, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,3)
plot(norm_freq, mod)
ylabel('|H(w)|, L=64')

L = 0:63
x = exp(j * w0 * L)
[H, W] = dtft(x.*w2, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,4)
plot(norm_freq, mod)
ylabel('|H(w)X(w)|')


[H, W] = dtft(w3, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,5)
plot(norm_freq, mod)
ylabel('|H(w)|, L=128')

L = 0:127
x = exp(j * w0 * L)
[H, W] = dtft(x.*w3, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,6)
plot(norm_freq, mod)
ylabel('|H(w)X(w)|')

[H, W] = dtft(w4, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,7)
plot(norm_freq, mod)
xlabel('Normalized Frequency') 
ylabel('|H(w)|. L=256')

L = 0:255
x = exp(j * w0 * L)
[H, W] = dtft(x.*w4, N)
mod = abs(H)
norm_freq = W ./ pi
subplot(4,2,8)
plot(norm_freq, mod)
xlabel('Normalized Frequency') 
ylabel('|H(w)X(w)|')


%% Apartado C

w = hann(32)'
plot_dtft(w, N)

N = 1024
n = 0:31
x = [ones(1, 32)]
plot_dtft(x, N)

plot_dtft(w.*exp(j*w0*n), N)
plot_dtft(x.*exp(j*w0*n), N)




