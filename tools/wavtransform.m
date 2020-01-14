function Y=wavtransform(TS,fq,sr,width,dim)
%              Y=wavtransform(TS,fq,sr,width,dim)
%It returns the morlet wavelet transform of 'TS' computed at frequencies
%'fq' [1:100], along the dimension 'dim' [the longest].
%TS is a timeseries or a matrix of timeseries sampled at rate 'sr' [250Hz] 
%'width' defines the number of cycles for the mother wavelet  [5], unless
%for  width=0 that calls cwt.
%NOTE: Width affects bandwith and the time/frequency resolution trade-off,
%i.e. with increasing width frequency resolution increases and time
%resolution decreases:
%   bandwith=fq/width*2;
%    wavelet duration=width/F/pi;
%        
%              Y=wavtransform(TS,fq,sr,width,dim)

% Nicola Polizotto (2020), UTHealth, Houston, Texas, USA



if ~nargin
    help wavtransform
    return
end


if ~exist('dim','var')||isempty(dim)
[ntp, nts]=size(TS);
if nts>ntp
    TS=TS';
    [ntp, nts]=size(TS);
end
else
    if dim~=1
        TS=TS';
    end
    [ntp, nts]=size(TS);
end



if ~exist('width','var')||isempty(width)
    width=5;
end
if ~exist('fq','var')||isempty(fq)
    fq=1:100;
end
if ~exist('sr','var')||isempty(sr)
    sr=250;
end


%load all Mother Wavelets
if any(width~=0)
dt = 1/sr;
MW=cell(numel(width),numel(fq));
for nw=1:numel(width)
for nf=1:numel(fq)
w=width(nw);  
sf = fq(nf)/w;
st = 1/(2*pi*sf); 
% t=-3.5*st:dt:3.5*st; 
t=-(w/2)*st:dt:(w/2)*st; 
MW(nw,nf) = {morwav(fq(nf),t,w)};
end
end
end
scales=sr./fq;

%initialize
% Y=complex(zeros(numel(fq),ntp,nts,numel(width)));
Y=zeros(numel(fq),ntp,nts,numel(width));



for nw=1:numel(width)
if width(nw)==0
for N=1:nts
Y(:,:,N,nw) = cwt(TS(:,N),scales,'cmor1-1'); 
end    
     
else
for N=1:nts
for nf=1:numel(fq)
Y(nf,:,N,nw) = conv(TS(:,N),MW{nw,nf}','same');
end
end
end
end
end



function y = morwav(f,t,width)
sf = f/width;
st = 1/(2*pi*sf);
A = 1/sqrt((st*sqrt(pi)));
y = A*exp(-t.^2/(2*st^2)).*exp(1i*2*pi*f.*t);
end

