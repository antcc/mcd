function [] = plot_dtft (h, N)

[H, W] = dtft(h, N)
mod = abs(H)
norm_freq = W ./ pi

figure;
subplot(2,1,1)
plot(norm_freq, mod)
title('Magnitude Response')
xlabel('Normalized Frequency') 
ylabel('|H(w)|')
grid on

subplot(2,1,2)
plot(norm_freq, rad2deg(angle(H)))
title('Phase Response')
xlabel('Normalized Frequency') 
ylabel('Degrees')
grid on
