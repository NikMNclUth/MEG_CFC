function [ts,wavout,verts] = thresholdROI(data,filtlow,filthigh,blwind,wind,sr,frq,filtdata)
%thresholdROI -thresholding for source localised MEG data. Narrows down
%region of interest to appropriate sources using 5 different methods. 
% method 1 - use 99th percentile of surrogate "noise" distribution to
% determine the minimum acceptable criterion for power.
% method 2 - method 1 but with 95th percentile.
% method 3 - find significant vertices by counting the number of surrogate
% values greater than each feature.
% method 4 - use normcdf of vertex features to find out probability of
% occurance
% method 5 - take the 95% percentile of the vertex wavelet features as the
% cut off for acceptable power.
%
% Inputs
% data = time series data (verts x time x trials)
% filtlow = low cut off for filter
% filthigh = high cut off for filter
% blwind = baseline period window
% wind = stim period window
% sr = sampling rate
% frq = frequency of interest
% filtdata = filter the data before final wavelets? (1 = yes, anything else
% = no)
%
% Outputs
% ts = structure with 5 time series (pre-wavelet transform time series for
% each method)
% wavout = structure with 5 wavelet transforms (1 for each method)
% verts = matrix of 1s and 0s corresponding to the vertices used in each
% method

% Nicholas Murphy (2020), Baylor College of Medicine, Houston, Texas, USA

%% prepare data
% filter

tempA = zeros(size(data,1),size(data,2)*size(data,3));
parfor iter = 1:size(data,1)
    ampdat = eegfilt(data(iter,:,:),sr,filtlow,filthigh,0,[],0,'fir1',0);
    tempA(iter,:) = ampdat;
end
clear ampdat


% tempA = eegfilt(data,sr,filtlow,filthigh,0,[],0,'fir1',0);
tempA = reshape(tempA,size(data,1),size(data,2),size(data,3));
% wavelet features
wavs = zeros(size(data,1),size(data,2));
for iter = 1:size(data,1)
    wavs(iter,:) = baseline(abs(wavtransform(nanmean(tempA(iter,:,:),3),frq,sr,10)),blwind,2,[]);
end
feats = mean(wavs(:,wind),2);
% feature based selection (top 5%)
cutoff = prctile(feats,95);
tfdat  = feats>cutoff;


%% create surrogates from baseline data
tempB = tempA(:,blwind,:);
tempB = mean(mean(tempB,1),3); % create "noise" timeseries to base surrogates on
[zM] = aaftSurr(tempB,1000); zM = zM';
wavS = zeros(1000,size(zM,2));
for iter = 1:size(zM,1)
    wavS(iter,:) = abs(wavtransform(zM(iter,:),frq,sr,10));
end
featS = mean(wavS,2);


%% find significant vertices
thresh = zeros(size(feats,1),4);
% method 1 - simple cut off (99%)
cutoff = prctile(featS,99);
thresh(:,1)=feats>cutoff;
% method 2 - simple cut off (95%)
cutoff = prctile(featS,95);
thresh(:,2)=feats>cutoff;
% method 3 - simple p value
for iter = 1:size(feats,1)
    f = numel(find(featS>feats(iter)))/1000;
    if f == 0
        f = 0.0000001;
    end
    if f == 1
        thresh(iter,3) = NaN;
    else
        thresh(iter,3) = f;
    end
    if thresh(iter,3)<0.01
        thresh(iter,3) = 1;
    else
        thresh(iter,3) = 0;
    end
end
% method 4 - normcdf p value
for iter = 1:size(feats,1)
    f = 1-normcdf(feats(iter));
    if f == 0
        thresh(iter,4) = NaN;
    else
        thresh(iter,4) = f;
    end
    if thresh(iter,4)<0.01
        thresh(iter,4) = 1;
    else
        thresh(iter,4) = 0;
    end
end
thresh = logical(thresh);
thresh(:,5) = tfdat;
%% Avoid NaNs
% some of the data might not be significantly greater than noise, avoid
% inclusion of nans or missing data by keeping all vertices.
s = zeros(1,5);
s(1) = sum(thresh(:,1));s(2) = sum(thresh(:,2));s(3) = sum(thresh(:,3));s(4) = sum(thresh(:,4));s(5) = sum(thresh(:,5));
f = find(s==0);
if ~isempty(f)
    for ii = 1:numel(f)
        thresh(:,f(ii)) = logical(ones(size(thresh,1),1));
    end
end
%% wavelet outputs

if filtdata == 1
    tp = size(data,2);tr = size(data,3);
    data = tempA;
%     data = reshape(data,size(data,1),tp,tr);
end
orig = mean(mean(data,3),1);
data1 = mean(mean(data(thresh(:,1),:,:),3),1);
data2 = mean(mean(data(thresh(:,2),:,:),3),1);
data3 = mean(mean(data(thresh(:,3),:,:),3),1);
data4 = mean(mean(data(thresh(:,4),:,:),3),1);
data5 = mean(mean(data(tfdat,:,:),3),1);
wavo = baseline(abs(wavtransform(orig,1:60,sr,10)),200:1000,2,[]);
wav1 = baseline(abs(wavtransform(data1,1:60,sr,10)),200:1000,2,[]);
wav2 = baseline(abs(wavtransform(data2,1:60,sr,10)),200:1000,2,[]);
wav3 = baseline(abs(wavtransform(data3,1:60,sr,10)),200:1000,2,[]);
wav4 = baseline(abs(wavtransform(data4,1:60,sr,10)),200:1000,2,[]);
wav5 = baseline(abs(wavtransform(data5,1:60,sr,10)),200:1000,2,[]);

%% organise outputs

ts.orig = orig;
ts.data1 = data1;
ts.data2 = data2;
ts.data3 = data3;
ts.data4 = data4;
ts.data5 = data5;
wavout.wavo = wavo;
wavout.wav1 = wav1;
wavout.wav2 = wav2;
wavout.wav3 = wav3;
wavout.wav4 = wav4;
wavout.wav5 = wav5;
verts = thresh;

end

function [zM] = aaftSurr(TS,surrs)
xV=TS;
n = length(xV);
zM = NaN*ones(n,surrs);
[oxV,T] = sort(xV);
[~,ixV] = sort(T);
for isur=1:surrs
    % Rank order a white noise time series 'wV' to match the ranks of 'xV'
    wV = randn(n,1) * std(xV);
    [owV,~]= sort(wV);
    yV = owV(ixV);
    % Fourier transform, phase randomization, inverse Fourier transform
    if rem(n,2) == 0
        n2 = n/2;
    else
        n2 = (n-1)/2;
    end
    tmpV = fft(yV,2*n2);
    magnV = abs(tmpV);
    fiV = angle(tmpV);
    rfiV = rand(n2-1,1)*2*pi;
    nfiV = [0; rfiV; fiV(n2+1); -flipud(rfiV)];
    tmpV = [magnV(1:n2+1)' flipud(magnV(2:n2))']';
    tmpV = tmpV .* exp(nfiV .* 1i);
    yftV=real(ifft(tmpV,n));  % Transform back to time domain
    % Rank order the 'xV' to match the ranks of the phase randomized time series
    [~,T2] = sort(yftV);
    [~,iyftV] = sort(T2);
    zM(:,isur) = oxV(iyftV);  % the AAFT surrogate of xV   
end
clear T2 iyftV yftV tmpV owV wV

end
