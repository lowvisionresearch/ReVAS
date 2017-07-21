function [PS,f] = ComputePowerSpectra(Fs,position, minLength, overlap, flag)
% this function computes the power spectral density of eye position traces
% it uses a sliding window of minLength and overlap specified by increment.
% e.g., if increment = minLength/4, it means 75% overlap. 
% e.g., if increment = minLength/2, the overlap is 50%

% remove NANs
position(isnan(position)) = [];

if overlap<0 || overlap>1
    overlap = 0.5;
    warning('overlap must be within 0 and 1.');
end

if nargin<5
    flag = 0;
end

increment = round(minLength*(1-overlap));
L = minLength; % modified to reflect number of samples rather than duration
if rem(L,2)~=0
    L = L-1;
end
howManyTimes = length(1:increment:(length(position)-L));
if howManyTimes == 0
    howManyTimes = 1;
end

PS = zeros(floor(L/2)+1, howManyTimes);


if flag
    wb = waitbar(0,'Computing PSD...');
    cols = jet(howManyTimes);
end

if flag==2
    fh=figure;
end

f = Fs*(0:(L/2))/L;
for i=1:howManyTimes
%     iterPosition = sqrt(sum(position((i-1)*increment+1:(i-1)*increment+L,:).^2,2));
    iterPosition = position((i-1)*increment+1:(i-1)*increment+L,1);
    Y = fft(iterPosition);
    P2 = abs(Y).^2/L;
    P1 = P2(1:(L/2+1));
    P1(2:end-1) = 2*P1(2:end-1);
    PS(:,i) = P1;
    
    if flag
        waitbar(i/howManyTimes,wb,sprintf('Computing PSD... %.1f',100*(i/howManyTimes)));
    end
    
    if flag==2
        figure(fh);
        semilogx(f,10*log10(PS(:,i)),'-','Color',cols(i,:)); hold on;
        xlim([.05 2000]);
        drawnow;
    end
end

PS = nanmean(10*log10(PS),2);


if flag
    delete(wb);
end