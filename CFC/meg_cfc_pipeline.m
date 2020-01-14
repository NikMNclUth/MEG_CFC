function [] = meg_cfc_pipeline(contents,vertexfile,vertind,namepatternR,namepatternL,datapath,amps,phases,wind,blwind,sr,entfrq,savepath,surrs)

%% Inputs

%contents = list of file names to load
%vertexfile = matrix with logical indices of the vertices to use
%vertind = column to use from vertexfile
%namepatternR/L = loading parameter for picking particular files in mixed folder
%datapath = path of the data
%amps = vector of amplitude frequencies to use
%phases = vector of phase frequencies to use
%wind = window to estimate pac
%blwind = window to estimate baseline pac
%sr = sampling rate
%entfrq = entrainment frequency (e.g. ASSR 40 Hz = 40)
%savepath = where to save the data
%surrs = number of surrogates to use

% Nicholas Murphy (2020), Baylor College of Medicine, Houston, Texas, USA

%% Function
% load and process

for iter = 1:length(contents)
    clearvars -except contents vertexfile vertind namepatternR namepatternL datapath amps phases wind ...
        blwind sr entfrq savepath surrs iter
    %% load subject data for left and right hemisphere
    nom = contents{iter,1}(1:3);
    disp(['processing ',nom]);
    load([datapath,nom,namepatternR]);
    load([datapath,nom,namepatternL]);
    eval(['stcr = permute(stcs',num2str(entfrq),'_anat_bothR,[2,3,1]);']);
    eval(['stcl = permute(stcs',num2str(entfrq),'_anat_bothL,[2,3,1]);']);
    %% load subject vertices
    eval(['vertsR = vertexfile.',nom,'.right.verts(:,',num2str(vertind),');']);
    eval(['vertsL = vertexfile.',nom,'.left.verts(:,',num2str(vertind),');']);
    % avoid NaNs in vertices
    s = zeros(1,2);
    s(1) = sum(vertsR);s(2) = sum(vertsL);
    f = find(s==0);
    if ~isempty(f)
        vertnames = {'vertsR','vertsL',};
        vertnames = vertnames(f);
        for ii = 1:numel(f)
            eval([vertnames{ii} '= logical(ones(size(',vertnames{ii},',1),1));']);
            
        end
    end
    stcr = nanmean(stcr(vertsR,:,:),1);
    stcl = nanmean(stcl(vertsL,:,:),1);
    
    %% Organise CFC essentials
    % remove broken data/NaNs/deal with unexpected surprises
    stcr(:,:,(sum(isnan(squeeze(stcr)),1)>0)) = [];
    stcl(:,:,(sum(isnan(squeeze(stcl)),1)>0)) = [];
    % get dimension sizes
    [~,tp,trr] = size(stcr);
    [~,~,trl] = size(stcl);
    % set indices for cfc
    ioi = false(tp,1);
    ioi(wind) = true;
    ioib = false(tp,1);
    ioib(blwind) = true;
    % create window to normalise time series amplitudes to
    normwind = false(tp,1);
    normwind(200:length(normwind)-200) = true; % exclude edges
    
    %% Filter the time series
   
    ampdataR = zeros(length(amps),size(stcr,2),size(stcr,3));
    ampdataL = zeros(length(amps),size(stcl,2),size(stcr,3));
    phdataR = zeros(length(phases),size(stcr,2),size(stcr,3));
    phdataL = zeros(length(phases),size(stcl,2),size(stcr,3));
    
    ampdataR(:,:,:) = reshape(abs(wavtransform(stcr(:,:),amps,sr,10)).^2,length(amps),size(stcr,2),size(stcr,3));
    phdataR(:,:,:) = reshape(angle(wavtransform(stcr(:,:),phases,sr,10)),length(phases),size(stcr,2),size(stcr,3));
    ampdataL(:,:,:) = reshape(abs(wavtransform(stcl(:,:),amps,sr,10)).^2,length(amps),size(stcl,2),size(stcl,3));
    phdataL(:,:,:) = reshape(angle(wavtransform(stcl(:,:),phases,sr,10)),length(phases),size(stcl,2),size(stcl,3));
    
    rdat = reshape(wavtransform(stcr(:,:),phases,sr,10),length(phases),size(stcr,2),size(stcr,3));
    ldat = reshape(wavtransform(stcl(:,:),phases,sr,10),length(phases),size(stcl,2),size(stcl,3));
    
    %     %% Non-Sinusoidal Waveforms ---- optional testing
%     [ratiosRs] = ns_rise_decay(abs(rdat(:,wind,:)));
%     [ratiosRb] = ns_rise_decay(abs(rdat(:,blwind,:)));
%     [ratiosLs] = ns_rise_decay(abs(ldat(:,wind,:)));
%     [ratiosLb] = ns_rise_decay(abs(ldat(:,blwind,:)));

    %     savename = [savepath,nom,num2str(entfrq),'risedecayratio.mat'];
    %     save(savename,'ratiosRs','ratiosRb','ratiosLs','ratiosLb');
    %     clear ratios*
    
    %% Create Surrogates
    
    [~,surrtemplaters] = randpermsurr(rdat,surrs,trr,2,wind);
    [~,surrtemplaterb] = randpermsurr(rdat,surrs,trr,2,blwind);
    [~,surrtemplatels] = randpermsurr(ldat,surrs,trl,2,wind);
    [~,surrtemplatelb] = randpermsurr(ldat,surrs,trl,2,blwind);
    
    
    %% Cross-Frequency Coupling
    Xr = zeros(length(amps),length(phases));
    Xrb = zeros(length(amps),length(phases));
    Xl = zeros(length(amps),length(phases));
    Xlb = zeros(length(amps),length(phases));
    
    % stimulus window with trial to trial amp norm
    [Xr(:,:),~]=cfcx2(phdataR(:,:,:),ampdataR(:,:,:),ioi,[],normwind);
    % baseline window with trial to trial amp norm
    [Xrb(:,:),~]=cfcx2(phdataR(:,:,:),ampdataR(:,:,:),ioib,[],normwind);
    % stimulus window with trial to trial amp norm
    [Xl(:,:),~]=cfcx2(phdataL(:,:,:),ampdataL(:,:,:),ioi,[],normwind);
    % baseline window with trial to trial amp norm
    [Xlb(:,:),~]=cfcx2(phdataL(:,:,:),ampdataL(:,:,:),ioib,[],normwind);
    % prep surrogate cfc
    Xrss = zeros(numel(amps),numel(phases),surrs);
    Xrsb = zeros(numel(amps),numel(phases),surrs);
    Xlss = zeros(numel(amps),numel(phases),surrs);
    Xlsb = zeros(numel(amps),numel(phases),surrs);
    
    parfor ii = 1:surrs
        % right hemisphere
        [Xrss(:,:,ii),~]=cfcx2(surrtemplaters(:,:,:,ii),ampdataR,ioi,[],normwind);
        [Xrsb(:,:,ii),~]=cfcx2(surrtemplaterb(:,:,:,ii),ampdataR,ioib,[],normwind);
        [Xlss(:,:,ii),~]=cfcx2(surrtemplatels(:,:,:,ii),ampdataL,ioi,[],normwind);
        [Xlsb(:,:,ii),~]=cfcx2(surrtemplatelb(:,:,:,ii),ampdataL,ioib,[],normwind);
        disp(num2str(ii))
    end
    
    %% Hypothesis Driven Z-Transform
    hdxrss = zeros(1,surrs);
    hdxrsb = zeros(1,surrs);
    hdxlss = zeros(1,surrs);
    hdxlsb = zeros(1,surrs);
    
    % find indices to use
    test=phases.*amps';
    hypphase = 6;
    hypamp = entfrq;
    f = find(phases==hypphase);
    f = find(phases==hypphase);
    ff = find(test(:,f)==hypphase*hypamp);
    for ii = 1:surrs
        hdxrss(:,ii) = squeeze(Xrss(ff,f,ii));
        hdxrsb(:,ii) = squeeze(Xrsb(ff,f,ii));
        hdxlss(:,ii) = squeeze(Xlss(ff,f,ii));
        hdxlsb(:,ii) = squeeze(Xlsb(ff,f,ii));
    end
    Zhypcfcrs = zeros(size(Xr));Zhypcfcrb = zeros(size(Xr));
    Zhypcfcls = zeros(size(Xr));Zhypcfclb = zeros(size(Xr));
    Zhypcfcrs = (Xr-mean(hdxrss))./std(hdxrss);
    Zhypcfcrb = (Xrb-mean(hdxrsb))./std(hdxrsb);
    Zhypcfcls = (Xl-mean(hdxlss))./std(hdxlss);
    Zhypcfclb = (Xlb-mean(hdxlsb))./std(hdxlsb);
    
    %% mapwise z-transform
    for ii = 1:size(Xrss,3)
        temp = Xrss(:,:,ii);
        temprsm(ii) = mean(temp(:));
        temprsd(ii) = std(temp(:));
        temp = Xrss(:,:,ii);
        temprbm(ii) = mean(temp(:));
        temprbd(ii) = std(temp(:));
    end
    for ii = 1:size(Xlss,3)
        temp = Xlss(:,:,ii);
        templsm(ii) = mean(temp(:));
        templsd(ii) = std(temp(:));
        temp = Xlss(:,:,ii);
        templbm(ii) = mean(temp(:));
        templbd(ii) = std(temp(:));
    end
    Zhypcfcrs2 = (Xr-mean(temprsm))./std(temprsm);
    Zhypcfcrb2 = (Xrb-mean(temprbm))./std(temprbm);
    Zhypcfcls2 = (Xl-mean(templsm))./std(templsm);
    Zhypcfclb2 = (Xlb-mean(templbm))./std(templbm);
    %% individual pairing z-transform
    for ii = 1:length(amps)
        for iv = 1:length(phases)
            Zhypcfcrs3(ii,iv) = (Xr(ii,iv)-mean(Xrss(ii,iv,:),3))./std(Xrss(ii,iv,:),[],3);
            Zhypcfcrb3(ii,iv) = (Xrb(ii,iv)-mean(Xrsb(ii,iv,:),3))./std(Xrsb(ii,iv,:),[],3);
            Zhypcfcls3(ii,iv) = (Xl(ii,iv)-mean(Xlss(ii,iv,:),3))./std(Xlss(ii,iv,:),[],3);
            Zhypcfclb3(ii,iv) = (Xlb(ii,iv)-mean(Xlsb(ii,iv,:),3))./std(Xlsb(ii,iv,:),[],3);
        end
    end
    
    % save the filtered data
    savename = [savepath,nom,num2str(entfrq),'cfc.mat'];
    save(savename,'Xr','Xrb','Xl','Xlb',...
        'Zhypcfcrs','Zhypcfcrb','Zhypcfcls','Zhypcfclb',...
        'Zhypcfcrs2','Zhypcfcrb2','Zhypcfcls2','Zhypcfclb2',...
        'Zhypcfcrs3','Zhypcfcrb3','Zhypcfcls3','Zhypcfclb3',...
        'hdxrss','hdxrsb','hdxlss','hdxlsb','-v7.3');
end
end


%% Subroutines


function [surrdata,surrtemplate] = randpermsurr(data,surrs,tr,wind)
xV = zeros(1,length(wind)*tr);
% go through filtered data and shuffle
surrdata = zeros(size(data,1),length(wind),tr,surrs);
temp = data(:,wind,:);
[xx,yy,zz] = size(temp);
temp = temp(:,:);
% get samples
randtime = randsample(round(length(xV)*.8),surrs)+round(length(xV)*.1);
parfor ii = 1:surrs
    temp2 = [temp(:,randtime(ii):end) temp(:,1:randtime(ii)-1)];
    temp2 = reshape(temp2,xx,yy,zz);
    temp2 = angle(temp2);
    surrdata(:,:,:,ii) = temp2;
end
surrtemplate = zeros(size(data,1),size(data,2), size(data,3),size(surrdata,4));
surrtemplate(:,wind,:,:) = surrdata;
end


function [ratios] = ns_rise_decay(phasedata)


%% Citation:
% Seymour, Robert A., Gina Rippon, and Klaus Kessler. 
% "The detection of phase amplitude coupling during sensory processing." Frontiers in neuroscience 11 (2017): 487.



%% code:
for ii = 1:size(phasedata,1)
    test = phasedata(ii,:,:);
    p=1;
    for iter = 1:size(phasedata,3)
        trl = test(:,:,iter);
        trlfl = trl.*-1;
        [~,peak_locations] = findpeaks(trl);
        [~,trough_locations] = findpeaks(trlfl);
        % Equalise the number of peak and trough events
        if ~isempty(peak_locations) && ~isempty(trough_locations)
            if length(peak_locations) > length(trough_locations)
                peak_locations(1) = [];
            elseif length(peak_locations) < length(trough_locations)
                trough_locations(1) = [];
            end
            
            % Calculate time to peak and time to decay
            time_to_decay = [];
            time_to_peak = [];
            if peak_locations(1)<trough_locations(1) %if peak first
                for i = 1:length(peak_locations)-1
                    time_to_decay(i) = trough_locations(i)-peak_locations(i);
                    time_to_decay_all(p) = trough_locations(i)-peak_locations(i);
                    time_to_peak(i) = abs(peak_locations(i+1)-trough_locations(i));
                    time_to_peak_all(p) = abs(peak_locations(i+1)-trough_locations(i));
                    p = p+1;
                end
            elseif peak_locations(1)>trough_locations(1) %if trough first
                for i = 1:length(peak_locations)-1
                    time_to_decay(i) = peak_locations(i)-trough_locations(i);
                    time_to_decay_all(p) = peak_locations(i)-trough_locations(i);
                    time_to_peak(i) = abs(trough_locations(i+1)-peak_locations(i));
                    time_to_peak_all(p) = abs(trough_locations(i+1)-peak_locations(i));
                    p = p+1;
                end
            end
            ratios(ii,iter) = mean(time_to_decay)./mean(time_to_peak);
        elseif ~isempty(peak_locations) || ~isempty(trough_locations)
            ratios(ii,iter) = 0;
            
        end
    end
end
f = isnan(ratios);
ratios(f) = 0;
end

function [X,LBL]=cfcx2(ph,amp,ioi,normalz,normwind)
%CFC, Phase-Amplitude modulation is calculated on extracted ph/amp ([phoi,ntp]=size(ph)):
%          [X,LBL]=cfcx(ph,amp,method,nbin,fPH,sr,ioi,normalz)
%According to the following methods [ALL]:

%3 MVL        mean vector length (Canolty)

%NOTE
%multitrial ph and amp are supported (fq X tp X tr)
%'normalz' make the ranges for each trial match [1]

[phoi,ntp,ntr]=size(ph);
ampoi=size(amp,1);

if ~exist('ioi','var')||isempty(ioi)
    ioi=true(1,ntp);
end
if ntr>1
    if ~exist('normalz','var')||isempty(normalz)
        normalz=true;
    end
    if normalz
        mn=repmat(min(amp(:,normwind,:),[],2),[1 ntp 1]);
        rng=repmat(range(amp(:,normwind,:),2),[1 ntp 1]);
        amp=eps+(amp-mn)./rng;
    end
    amp_old=amp;
    amp=resh(amp(:,ioi,:));
    ph=resh(ph(:,ioi,:));
    ioi=repmat(ioi,[1 ntr]);
else
    amp_old=amp;
    amp=amp(:,ioi);
    ph=ph(:,ioi);
end

%MVL    Canolty
for nph=1:phoi
    X(:,nph)=abs(mean(amp.*exp(1i*repmat(ph(nph,:),[ampoi 1])),2)) ;
end


end
