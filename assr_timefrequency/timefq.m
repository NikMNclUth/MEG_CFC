function [IND,EV,PLF,TW,t2,fq2,X,TWC]=timefq(TS,fq,sr,width,dim, fixboundaries,lowres_t, lowres_fq, lowres_kind,cfcwindow,t_erpdimmer)
%   [IND,EV,PLF,TW,t2,fq2,X,TWC]=timefq(TS,fq,sr,width,dim, fixboundaries,lowres_t, lowres_fq, lowres_kind,cfcwindow,t_erpdimmer)
%TF analysis for observations repeated along dimension 'dim':
%-  IND/EV/PLF are computed averaging along 'dim'
%-  'fixboundaries' ~=0  to replace with NaN timepoints with not reliable
%estimates due to wavelet boundaries. Note that this allows computing
%baseline by simply averaging from zero.
%-  TW is  trialwise information. In order to have it in a manageable form
%for storage or analysis (history effects, within session differences,
%etc.), information can be retained in a low resolution form by averaging
%across time and frequency. Compression is controlled by 'lowres_t ' and
%'lowres_fq', so that it can can be tuned depending on the the needs
%(exploratory vs specific frequencies and timepoints). 'lowres_kind' can be
%'tot','ind', for low resolution version of total or induced activity
%(=total-evoked at full resolution).  't2' and 'fq2' are indices to map
%compressed form to input time and frequencies.
% lowres_t is the smoothing/decimation factor for time [40].
%          BUT    0   collapses completely time 
% lowres_fq is the smoothing/decimation factor for fq [-1]
%          BUT    0    collapses completely fqs 
%                   F<0  calls frequency  to band transformation
%NOTE: 
%-if lowres_kind is 'tc' thewhole, full resolution total complex form is produced
%-that can be output qurying the TWC output argument
%-4D complex input support. Instead of raw data, TS can be a complex form
%input. But note that instead of 'dim' input  EVOKED power
%should be provided! This allows quick computation of measures from imported wavelets
%output without need to recompute transformations. 
%  [IND,EV,PLF,TW,t2,fq2,X,TWC]=timefq(TS,fq,sr,width,dim, fixboundaries,lowres_t, lowres_fq, lowres_kind,cfcwindow,t_erpdimmer)

%Nicola Polizotto (2020), UTHealth, Houston, Texas, USA

if ~nargin
    help timefq
else
if ~exist('width','var')||isempty(width)
    width=[];
end
if ~exist('sr','var')||isempty(sr)
    sr=250;
end
if ~exist('fq','var')||isempty(fq)
    fq=1:100;
end
if ~exist('fixboundaries','var')||isempty(fixboundaries)
    fixboundaries=true;
end

lowres=false;
if exist('lowres_kind','var') && ~isempty(lowres_kind)
lowres=true;
if ~exist('lowres_fq','var')||isempty(lowres_fq)
    lowres_fq=-1;
end
if ~exist('lowres_t','var')||isempty(lowres_t)
    lowres_t=40;
end
end


if nargout>7
    twc=true;
else
    twc=false;
end

if nargout>6
    pac=true;
    fPH=fq<14;
    fA=fq>13;    
    
else
    pac=false;
end


d=size(TS);
if any(any(imag(TS(:,:,1,1))~=0)) %complex input
    iscomplex=true;
    EV=dim;  %to handle complex form I need EVOKED as an input!
    if size(TS,4)>1 %timefq 3D output?
        if d(1)==size(EV,1) %3D output, FTtr
            EV=dim;
        else
            warning('Input appears to be 4D complex but it cannot be intepreted!')
            return
        end   
    else
    end
else
iscomplex=false;
if d(1)==1
    TS=squeeze(TS);
    d=size(TS);
   if exist('dim','var') && ~isempty(dim)
     dim=dim-1;
   end
end
if ~exist('dim','var')||isempty(dim)
if numel(d)==3
    dim=3;
else
    if d(1)>d(2)
    dim=2;
    else
    dim=1;
    end
end
end

dimerp=0;
if exist('t_erpdimmer','var') && ~isempty(t_erpdimmer)
 %t_erpdimmer is taken to be an evoked power dimmer
dimerp=1;
minmax=[min(t_erpdimmer) max(t_erpdimmer)];
t_erpdimmer=wfilter(t_erpdimmer,100);
t_erpdimmer=transf(t_erpdimmer,'R');
t_erpdimmer(t_erpdimmer>(max(t_erpdimmer)*.4))=max(t_erpdimmer)*.4;
t_erpdimmer=wfilter(t_erpdimmer,20);
t_erpdimmer=transf(t_erpdimmer,'R');
t_erpdimmer=minmax(1)+t_erpdimmer*(minmax(2)-minmax(1));
end
end

if numel(d)==3 || iscomplex
if ~iscomplex
dim3=d(3);
if dim==3
    TS=reshape(TS,d(1),[])';
    TS=reshape(TS,d(2),d(3),[]);
    dim=2;
    d=size(TS);
end
if dim==2
EV=nan(numel(fq),d(1),d(3));  %Fq x T x E
IND=nan(numel(fq),d(1),d(3));
if nargout>2
PLF=nan(numel(fq),d(1),d(3));
end  
else
EV=nan(numel(fq),d(2),d(3));
IND=nan(numel(fq),d(2),d(3));
if nargout>2
PLF=nan(numel(fq),d(2),d(3));
end  
end
d=size(EV);
display(['3D  TF : ' num2str(d(1)) ' fq x ' num2str(d(2)) ' time points x ' num2str(d(3)) ' sources, across '   num2str(dim3) ' epochs' ])
else %initialize output for complex input
d=[size(TS,1) size(TS,2) size(TS,4)];
dim3=size(TS,3);
EV=dim;  
IND=nan(size(EV));%Fq x T x E
if nargout>2
PLF=nan(size(EV));
end   
display(['3D  TF from complex 4D input : ' num2str(d(1)) ' fq x ' num2str(d(2)) ' time points x ' num2str(d(3)) ' sources, across '   num2str(dim3) ' epochs' ])    
end


for n=1:d(3) %for each electrode
 if iscomplex
     TC=TS(:,:,:,n);
     %TOTAL 
     T=nanmean(abs(TC),3);
 else
ts=TS(:,:,n);
%EVOKED
if any(~isnan(ts(:)))
ERP=nanmean(ts,dim);
if dimerp
   ERP=(ERP+fqfilter(ERP,30,sr,'low',2))./2;  
   EVt=abs(wavtransform(ERP,fq,sr,width));
   EV(:,:,n)=EVt.*repmat(t_erpdimmer,size(EVt,1),1);
else
EV(:,:,n)=abs(wavtransform(ERP,fq,sr,width));
end
%TOTAL COMPLEX
if ~exist('tpdim','var')
tp=size(ERP); tp=max(tp);
tpdim=find(size(ts)==tp);
end
TC=wavtransform(ts,fq,sr,width,tpdim);
%TOTAL 
T=nanmean(abs(TC),3);
else
T=nan(size(EV(:,:,n)));   
end
end
 
%INDUCED
IND(:,:,n)=T-EV(:,:,n);
%PLF
if nargout>2
if any(~isnan(ts(:)))    
PLF(:,:,n)=abs(nanmean(TC./abs(TC),3));    
end
end    

if twc
       if n==1  
        display('including twc...')              
        TWC=TC;
        TWC(:,:,:,d(3))=0;        
        else
        TWC(:,:,:,n)=TC;
        end
else
    TWC=[];
end

if pac
    if n==1
        display(['including cfc for ' num2str(size(cfcwindow,1)) ' time window(s)...'])           
    end
    for ncfcwin=1:size(cfcwindow,1)
      cfcwin=cfcwindow(ncfcwin,:);
      PH=angle(TC(fPH,:,:));
      AMP=abs(TC(fA,:,:));        
      ph=PH(:,cfcwin,:);
      amp=AMP(:,cfcwin,:);      
       if n==1    
        X=cfcx(ph,amp,[1 3 5],15,fPH,sr);   
        X(:,:,:,d(3),ncfcwin)=0;
       else    
        X(:,:,:,n,ncfcwin)=cfcx(ph,amp,[1 3 5],15,fPH,sr);
       end
    end
end


if lowres 
if strcmpi(lowres_kind,'tc') %just output TC
        if n==1   
        display('including tc out...')                          
        t2=1:size(TC,2);
        fq2=fq;
        TW=TC;
        TW(:,:,:,d(3))=0;        
        else
        TW(:,:,:,n)=TC;
        end
       
else
if strcmpi(lowres_kind,'ind') %compressing induced
       lowresin=abs(TC)-repmat(EV(:,:,n), [1 1 size(TC,3)]); 
else %total power
       lowresin=abs(TC);  
end
lowresin=fixbound(lowresin,fq,width,sr);
if n==1
   display('including lowres ind/tc...')                              
   [TW,t2,fq2]=compress(lowresin,lowres_t, lowres_fq,fq);
   TW(:,:,:,d(3))=0;
else
   TW(:,:,:,n)=compress(lowresin,lowres_t, lowres_fq,fq);
end
end
else
    TW=[];
    t2=[];
    fq2=[];
end


if n>1
for nc=1:numel(num2str(n-1));
   fprintf('\b') 
end
end
fprintf('%s',num2str(n))
end
fprintf(' done.\n')







elseif numel(d)==2 %2D input
%EVOKED
ERP=nanmean(TS,dim);
if dimerp
   ERP=(ERP+fqfilter(ERP,30,sr,'low',2))./2;  
   EVt=abs(wavtransform(ERP,fq,sr,width));
   EV=EVt.*repmat(time,size(EVt,1),1);
else
EV=abs(wavtransform(ERP,fq,sr,width));
end
%TOTAL COMPLEX
TC=wavtransform(TS,fq,sr,width);
%TOTAL 
T=nanmean(abs(TC),3);
%INDUCED
IND=T-EV;
%PLF
if nargout>2
PLF=abs(nanmean(TC./abs(TC),3));    
end


if lowres 
if strcmpi(lowres_kind,'tc') %just output TC
       TW=TC;    
       t2=1:size(TW,2);
       fq2=fq;
else
if strcmpi(lowres_kind,'ind')
       lowresin=abs(TC)-repmat(EV, [1 1 size(TC,3)]); %induced
else
       lowresin=abs(TC);  
end
   [TW,t2,fq2]=compress(lowresin,lowres_t, lowres_fq,fq);
end
end


end

if fixboundaries
    IND=fixbound(IND,fq,width,sr);
    EV=fixbound(EV,fq,width,sr);
    if nargout>2
        PLF=fixbound(PLF,fq,width,sr);
    end
end

end


function M=fixbound(M,fq,width,sr)

if isempty(width) || width==0 
    width=1;
end

tp=size(M,2);

w=ceil(width*sr./fq);
exceedingtp=w>tp;
if any(exceedingtp)
    display(['NO good estimates for ' num2str(fq(exceedingtp)) 'Hz !'])
    w(exceedingtp)=tp;
end

mw=max(w);
nf=numel(fq);
W=false(nf,mw);
for n=1:nf
   W(n,1:w(n))=true;
end

i=1:mw;
A=M(:,i,:,:);
A(repmat(W,[1 1 size(A,3) size(A,4)]))=NaN;
M(:,i,:,:)=A;

i=size(M,2)-mw+1 : size(M,2);
W=fliplr(W);
A=M(:,i,:,:);
A(repmat(W,[1 1 size(A,3) size(A,4)]))=NaN;
M(:,i,:,:)=A;



function [M,t2,fq2]=compress(M,lowres_t,lowres_fq,fq)

if lowres_t==0 %T specifier
   M=nanmean(M,2);           
   t2=NaN;
elseif lowres_t>1
   [M,t2]=nanwfilter(M,lowres_t,2,1,1);       
else
    t2=1:size(M,2);
end
   
if lowres_fq==0 %F specifier
   M=nanmean(M,1);
   fq2=NaN;
elseif lowres_fq>1
   [M,fq2]=nanwfilter(M,lowres_fq,1,1,1);          
elseif lowres_fq<0
   [M,~,fq2]=f2b(M,fq,1);
elseif lowres_fq==1
    fq2=fq;   
end  









function   [OUT,LBL,centralFq]=f2b(IN,fq,dim,peakvalue)
%      [OUT,LBL,centralFq]=f2b(IN,fq,dim,peakvalue)
%frequencies to band of interest...
%D,Th,A,B1,B2,G1,G2  <  4,8,13,18,30,48,"100"
%'peakvalue' calls for peak value in the band instead of mean value [0]


LBL{1}='DELTA <4';
LBL{2}='THETA <8';
LBL{3}='ALFA <13';
LBL{4}='BETA1 <18';
LBL{5}='BETA2 <30';
LBL{6}='GAMMA1 <48';
LBL{7}='GAMMA2';

if ~nargin
if ~nargout
    help f2b
end
    OUT=LBL;
else

if ~exist('fq','var')||isempty(fq)
    fq=1:size(IN,1);
end
    
    
nfq=numel(fq);



% bandup=[4 8 13 18 30 100];
bandup=[4 8 14 18 30 48 100];

bandup=[0 bandup 1000];
blim_min=find(bandup<min(fq), 1,'last');
if isempty(blim_min)
   blim_min=1; 
end
blim_max=find(bandup>max(fq), 1,'first');
bandup=bandup(blim_min:blim_max);

[~,B]=histc(fq,bandup);
B=B+1;

B(fq>57 & fq<63)=NaN; B(fq>80)=NaN;
band=unique(B(B>0));
LBL=LBL(blim_min-2+band);

mB=min(B(B>0));
B=B-mB+1;
mB=max(B);
B(find(B==mB,1,'last')+1:end)=NaN;


if ~exist('dim','var') || isempty(dim)
    dim=size(IN);
%     dim=dim(1:2);
    dim=find(dim==nfq);
end
if ~exist('peakvalue','var') || isempty(peakvalue)
    peakvalue=false;
end

if ~peakvalue
if dim==1
OUT=zeros(mB,size(IN,2),size(IN,3),size(IN,4),size(IN,5),size(IN,6),size(IN,7));
for n=1:mB
OUT(n,:,:,:,:,:,:)=nanmean(IN(B==n,:,:,:,:,:),1);
end
elseif dim==2
OUT=zeros(size(IN,1),mB,size(IN,3),size(IN,4),size(IN,5),size(IN,6),size(IN,7));
for n=1:mB
OUT(:,n,:,:,:,:,:)=nanmean(IN(:,B==n,:,:,:,:,:),2);
end
elseif dim==3
OUT=zeros(size(IN,1),size(IN,2),mB,size(IN,4),size(IN,5),size(IN,6),size(IN,7));
for n=1:mB
OUT(:,:,n,:,:,:,:)=nanmean(IN(:,:,B==n,:,:,:,:),3);
end
end

else
if dim==1
OUT=zeros(mB,size(IN,2),size(IN,3),size(IN,4),size(IN,5),size(IN,6),size(IN,7));
for n=1:mB
OUT(n,:,:,:,:,:,:)=max(IN(B==n,:,:,:,:,:),[],1);
end
elseif dim==2
OUT=zeros(size(IN,1),mB,size(IN,3),size(IN,4),size(IN,5),size(IN,6),size(IN,7));
for n=1:mB
OUT(:,n,:,:,:,:,:)=max(IN(:,B==n,:,:,:,:,:),[],2);
end
elseif dim==3
OUT=zeros(size(IN,1),size(IN,2),mB,size(IN,4),size(IN,5),size(IN,6),size(IN,7));
for n=1:mB
OUT(:,:,n,:,:,:,:)=max(IN(:,:,B==n,:,:,:,:),[],3);
end  
end
end
    
    


if numel(dim)>1
    error('dimensions interpration is ambigous! (add ''dim'' argument)')    
elseif isempty(dim)
    error('fq vectors does not match input dimensions!')
elseif dim>3
    error('frequency dim beyond 3rd is not supported yet!')    
end


if nargout>2
    bandup2=[1 bandup(1:end-1)];
    centralFq=mean([bandup2;bandup]);
    centralFq=centralFq(band);
end
end







