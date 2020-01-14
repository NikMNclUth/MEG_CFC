function [YF,t]=fqfilter(Y,f,sr,method,dim,order)
%General form for frequency filtering:
%           [YF,t]=fqfilter(Y,f,sr,method,order, dim)
%Butterworth for high/low-pass, ideal filtering for notch.
%  'y'is the input timeseries 
%  'f' is/are the cut-off frequency /frequencies    [0.1]
%  'sr' is sample rate  [250]
%  'method' can be  highpass,lowpass,pass or notch    ['lowpass', 'pass' if 2 f are entered]
% 'order' is the order of the filter               [10]
%  'dim' codes time direction and overrides default   [longer dimension is time]
%            [YF,t]=fqfilter(Y,f,sr,method,dim,order)

% Nicola Polizotto (2020), UTHealth, Houston, Texas, USA


if ~nargin
    help fqfilter
    return
end

    
    

if ~exist('order','var') || isempty(order)
     order=10;
end
if ~exist('sr','var')||isempty(sr)
%      sr=1/2.5;
     sr=250;
end
if ~exist('f','var') || isempty(f)
     f=0.1;
end
if ~exist('method','var') || isempty(method)
    method='low';
end
if isa(Y,'single')
    issingle=1;
    Y=double(Y);
else
    issingle=0;
end

if ischar(f)
    f=lower(f);
    switch f
        case 'delta'
           f=4;
           if strcmpi(method(1),'N')
               method='high';
           else
               method='low';               
           end
         case 'theta'
           f=[4.5 7.5];
         case 'alfa'
           f=[8.5 12.5];  
         case 'beta'
           f=[13.5 29];   
         case 'gamma'
           f=[30 50];            
    end
end


if numel(f)==2
    f=sort(f);
    if any(isnan(f)) || any(isinf(f)) ||f(1)==0
        
        if isnan(f(1)) || f(1)==0
        method='low';
        else
        method='high';            
        end
        f=f(~isnan(f) & ~isinf(f) & f~=0);
    else
        if ~exist('method','var') || ~strcmpi(method(1),'N')
               method='pass';
        end
    end
end
    
      
    
    
    
    
    
flip=false;   
[d1, d2, d3]=size(Y);
YF=zeros(size(Y));
for d=1:d3 
y=Y(:,:,d);    
%filtfilt works columnwise and idealfiter is set to work columnwise: is d1 time?     
if exist('dim','var') && ~isempty(dim) 
    if dim==2
       flip=true;
    end
else
  if d1<d2
    flip=true;
  end  
end

if flip
   y=y';
   [d1, d2]=size(y);
end
    


% display([method ' - ' num2str([f order sr])])      
if  ~strcmpi(method(1),'N') && ~strcmpi(method(1),'P')
%low/high-pass 
fN = f/sr*2;
if strcmpi(method(1),'L')
    [b,a] = butter(order, fN, 'low');
elseif  strcmpi(method(1),'H')
    [b,a] = butter(order, fN, 'high');    
end
    yf= filtfilt(b, a, y);
else
%notch/passband    
if strcmpi(method(1),'N')
    method = 'notch';
else
    method = 'pass';
end
ts1=timeseries(y,1/sr:1/sr:d1*(1/sr));
ts2 = idealfilter(ts1,f,method);
yf= squeeze(ts2.Data) + repmat(mean(ts1),d1,1);
end

if flip
    yf=yf';
end

YF(:,:,d)=yf;
end
    
if issingle
    YF=single(YF);
end

if nargout==2
    t=1:d1;
    t=(t-1)*(1/sr);
end 

end



