clear;
close all;
lambda = 1064e-6;
k = 2 * pi / lambda;
dx = 12.5e-3;
pm = 501;
pn = 501;
lx = pm * dx;%长度
ly = pn * dx;
z1 = 125;
z2 = 80;
z3 = 150;

frequency_x = (([0 : pm - 1] - (pm - 1) / 2) /lx).';
frequency_y = (([0 : pn - 1] - (pn - 1) / 2) /ly).';
[frequency_gridx,frequency_gridy] = meshgrid(frequency_y,frequency_x);
H1 = exp(1i * k * z1 * sqrt(ones(pm,pn) - (lambda * frequency_gridx).^2 - (lambda * frequency_gridy).^2));
H1back = exp(-1i * k * z1 * sqrt(ones(pm,pn) - (lambda * frequency_gridx).^2 - (lambda * frequency_gridy).^2));
H2 = exp(1i * k * z2 * sqrt(ones(pm,pn) - (lambda * frequency_gridx).^2 - (lambda * frequency_gridy).^2));
H2back = exp(-1i * k * z2 * sqrt(ones(pm,pn) - (lambda * frequency_gridx).^2 - (lambda * frequency_gridy).^2));
H3 = exp(1i * k * z3 * sqrt(ones(pm,pn) - (lambda * frequency_gridx).^2 - (lambda * frequency_gridy).^2));
H3back = exp(-1i * k * z3 * sqrt(ones(pm,pn) - (lambda * frequency_gridx).^2 - (lambda * frequency_gridy).^2));

%%
A1 = Fx_gaussianbeam(pn,pm,1.5,dx);
figure;
mesh(abs(A1))
B1k = double(rgb2gray(imread('Y.jpg')))/255*(-1)+1;
B1 = zeros(pm,pn);
B1(101:400,101:400) = B1k;
B2k = double(rgb2gray(imread('X.jpg')))/255*(-1)+1;
B2 = zeros(pm,pn);
B2(101:400,101:400) = B2k;
B1 = smoothdata(B1,1,"gaussian",10);
B1 = smoothdata(B1,2,"gaussian",10);
B2 = smoothdata(B2,1,"gaussian",10);
B2 = smoothdata(B2,2,"gaussian",10);

Energyin = sum(A1.^2,"all");
Energyout1 = sum(B1.^2,"all");
Energyout2 = sum(B2.^2,"all");
B1  = B1 / sqrt(Energyout1 / Energyin) * 1.0;
B2  = B2 / sqrt(Energyout2 / Energyin) * 1.0;
%%
weight2 = zeros(pm,pn);
weight2((B2 > 0)) = 1;
weight2 = weight2 + 0.1;
ap1 = zeros(pm,pn);
ap1((B1 > 0.0)) = 1;
ap2 = zeros(pm,pn);
ap2((B2 > 0)) = 1;
%%
PHI1 = zeros(pm,pn);
PHI2 = zeros(pm,pn);
%%
A1 = gpuArray(A1);
PHI1 = gpuArray(PHI1);
H1 = gpuArray(H1);
H2 = gpuArray(H2);
H3 = gpuArray(H3);
H1back = gpuArray(H1back);
H2back = gpuArray(H2back);
H3back = gpuArray(H3back);
weight2 = gpuArray(weight2);

LR1 = 300e-3;
LR2 = 300e-3;
veloc1 = zeros(pm,pn);
veloc2 = zeros(pm,pn);
momentum = 0.992;
tic
for ii = 1:1200
    C = A1 .* exp(1i * PHI1);
    D = ifft2(ifftshift(fftshift(fft2(C)) .* H1));
    E = D .* exp(1i * PHI2);
    F = ifft2(ifftshift(fftshift(fft2(E)) .* H2));
    M = F .* ap1;
    G = ifft2(ifftshift(fftshift(fft2(M)) .* H3));
    L = abs(G).^2;
    I = weight2 .* (L - B2.^2).^2;
    costF(ii) = sum(sum(I));

    
    Lbar = 2 * weight2 .* (L - B2.^2);
    Gbar = 2 * G .* Lbar;
    Mbar = ifft2(ifftshift(fftshift(fft2(Gbar)) .* H3back));
    Fbar = Mbar .* ap1;
    Ebar = ifft2(ifftshift(fftshift(fft2(Fbar)) .* H2back));
    PHI2bar = imag(Ebar .* conj(E));
    Dbar = Ebar .* exp(-1i * PHI2);
    Cbar = ifft2(ifftshift(fftshift(fft2(Dbar)) .* H1back));
    PHI1bar = imag(Cbar .* conj(C));

    fang1 = - PHI1bar + momentum * veloc1;
    fang2 = - PHI2bar + momentum * veloc2;
    
    PHI1 = PHI1 + LR1 * fang1;
    PHI2 = PHI2 + LR1 * fang2;
    if ii > 10
        veloc1 = fang1;
        veloc2 = fang2;
    end
    % if ii == 10
    %     costF = gather(costF);
    % end
    
    % [ratio(ii), RMSE(ii), Fidelity(ii)] = Fx_evaluation(G,B2,A1);
   
    % ii
end
toc

%%
C = A1 .* exp(1i * PHI1);
D = ifft2(ifftshift(fftshift(fft2(C)) .* H1));
E = D .* exp(1i * PHI2);
F = ifft2(ifftshift(fftshift(fft2(E)) .* H2));
wen = F;
F = F .* ap1;
G = ifft2(ifftshift(fftshift(fft2(F)) .* H3));
J = abs(F).^2;
L = abs(G).^2;

figure;
imagesc(angle(exp(1i * PHI1)))
colormap(othercolor('BuOr_12'))
figure;
imagesc(angle(exp(1i * PHI2)))
colormap(othercolor('BuOr_12'))
figure;
imagesc(abs(D).^2)
colormap(othercolor('PuBu9'))
figure;
imagesc(L)
colormap(othercolor('PuBu9'))
figure;
imagesc(abs(wen).^2)
colormap(othercolor('PuBu9'))
figure;
imagesc(PHI1bar)
colormap(othercolor('BuOr_12'))
figure;
imagesc(PHI2bar)
colormap(othercolor('BuOr_12'))

Fx_evaluation(G,B2,A1)
