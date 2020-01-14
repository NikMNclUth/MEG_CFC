function [timef_output,tffeatsRight,tffeatsLeft] = meg_feat_extract_TIMEF(datapath,namepatternR,namepatternL,contents,sr,wind,blwind,entfrq,vertexfile,savepath,fname)
% meg_feat_extract_TIMEF - extract the roi thresholded time-frequency data
% for psychcfc
%
% Inputs
% datapath = location of MNE processed mat data
% namepatternR/L = MNE processed mat data filename
% contents = directory contents list (files as text)
% sr = sampling rate
% wind = stimulus window
% blwind = baselinewind
% entfrq = entrainment frequency
% vertexfile = variable containing vertices for thresholding
% savepath = location to store processed data
% fname = identifier for group/condition data e.g. 'H40' - must be text
%
% Output
% timef_output = group data for timefrequency (structure format containing
% left and right evoked, induced, and plf data for methods 1 and 2
% (equivalent to methods 2 and 5 in the threshold function)
% tffeatsRight/Left = table with power/plf extracted across frequencies and
% time.

% Nicholas Murphy (2020), Baylor College of Medicine, Houston, Texas, USA

%% Storgage
% image data
wavR1 = zeros(60,3001,size(contents,1));
wavR2 = zeros(60,3001,size(contents,1));
wavL1 = zeros(60,3001,size(contents,1));
wavL2 = zeros(60,3001,size(contents,1));
plfR1 = zeros(60,3001,size(contents,1));
plfR2 = zeros(60,3001,size(contents,1));
plfL1 = zeros(60,3001,size(contents,1));
plfL2 = zeros(60,3001,size(contents,1));
indR1 = zeros(60,3001,size(contents,1));
indR2 = zeros(60,3001,size(contents,1));
indL1 = zeros(60,3001,size(contents,1));
indL2 = zeros(60,3001,size(contents,1));
% feature data
EpowR = zeros(length(contents),2); % col 1 = method1, col2 = method2
EpowL = zeros(length(contents),2); % col 1 = method1, col2 = method2
IpowR = zeros(length(contents),2); % col 1 = method1, col2 = method2
IpowL = zeros(length(contents),2); % col 1 = method1, col2 = method2
plfR = zeros(length(contents),2); % col 1 = method1, col2 = method2
plfL = zeros(length(contents),2); % col 1 = method1, col2 = method2


%% Image Loop

for iter = 1:length(contents)
    nom = contents{iter,1}(1:3);
    nomout{iter,1} = nom;
    disp(['processing ',nom]);
    load([datapath,nom,namepatternR]);
    load([datapath,nom,namepatternL]);
    eval(['stcr = permute(stcs',num2str(entfrq),'_anat_bothR,[2,3,1]);']);
    eval(['stcl = permute(stcs',num2str(entfrq),'_anat_bothL,[2,3,1]);']);
    eval(['vertsR1 = vertexfile.',nom,'.right.verts(:,2);']);
    eval(['vertsL1 = vertexfile.',nom,'.left.verts(:,2);']);
    eval(['vertsR2 = vertexfile.',nom,'.right.verts(:,5);']);
    eval(['vertsL2 = vertexfile.',nom,'.left.verts(:,5);']);
    stcr1 = mean(stcr(vertsR1,:,:),1);
    stcr2 = mean(stcr(vertsR2,:,:),1);
    stcl1 = mean(stcl(vertsL1,:,:),1);
    stcl2 = mean(stcl(vertsL2,:,:),1);
    % get wavelet and plf features
    [INDr1,EVr1,plfR1(:,:,iter)]=timefq(nanmean(stcr1,1),1:60,sr,10,3,0,[],[],[],[],[]);
    [INDr2,EVr2,plfR2(:,:,iter)]=timefq(nanmean(stcr2,1),1:60,sr,10,3,0,[],[],[],[],[]);
    [INDl1,EVl1,plfL1(:,:,iter)]=timefq(nanmean(stcl1,1),1:60,sr,10,3,0,[],[],[],[],[]);
    [INDl2,EVl2,plfL2(:,:,iter)]=timefq(nanmean(stcl2,1),1:60,sr,10,3,0,[],[],[],[],[]);    
    % baseline correct evoked and induced power
    for ii = 1:2
        eval(['wavR',num2str(ii),'(:,:,iter)= baseline(EVr',num2str(ii),',blwind,2,[]);']);
        eval(['wavL',num2str(ii),'(:,:,iter)= baseline(EVl',num2str(ii),',blwind,2,[]);']);
        eval(['indR',num2str(ii),'(:,:,iter)= baseline(INDr',num2str(ii),',blwind,2,[]);']);
        eval(['indL',num2str(ii),'(:,:,iter)= baseline(INDl',num2str(ii),',blwind,2,[]);']);        
    end
    clear INDr1 EVr1 INDr2 EVr2 INDl1 EVl1 INDl2 EVl2 stcr1 stcr2 stcl1 stcl2 vertsR1 vertsR2 vertsL1 vertsL2 stcr stcl nom
end
timef_output.right.evoked1 = wavR1;
timef_output.right.evoked2 = wavR2;
timef_output.right.induced1 = indR1;
timef_output.right.induced2 = indR2;
timef_output.right.plf1 = plfR1;
timef_output.right.plf2 = plfR2;
timef_output.left.evoked1 = wavL1;
timef_output.left.evoked2 = wavL2;
timef_output.left.induced1 = indL1;
timef_output.left.induced2 = indL2;
timef_output.left.plf1 = plfL1;
timef_output.left.plf2 = plfL2;
%% Feature Extraction
frange = entfrq-5:entfrq+5;
EpowR(:,1) = nanmean(squeeze(nanmean(wavR1(frange,wind,:),1)),1);
EpowR(:,2) = nanmean(squeeze(nanmean(wavR2(frange,wind,:),1)),1);
EpowL(:,1) = nanmean(squeeze(nanmean(wavL1(frange,wind,:),1)),1);
EpowL(:,2) = nanmean(squeeze(nanmean(wavL2(frange,wind,:),1)),1);
plfR(:,1) = nanmean(squeeze(nanmean(plfR1(entfrq,wind,:),1)),1);
plfR(:,2) = nanmean(squeeze(nanmean(plfR2(entfrq,wind,:),1)),1);
plfL(:,1) = nanmean(squeeze(nanmean(plfL1(entfrq,wind,:),1)),1);
plfL(:,2) = nanmean(squeeze(nanmean(plfL2(entfrq,wind,:),1)),1);
IpowR(:,1) = nanmean(squeeze(nanmean(indR1(frange,wind,:),1)),1);
IpowR(:,2) = nanmean(squeeze(nanmean(indR2(frange,wind,:),1)),1);
IpowL(:,1) = nanmean(squeeze(nanmean(indL1(frange,wind,:),1)),1);
IpowL(:,2) = nanmean(squeeze(nanmean(indL2(frange,wind,:),1)),1);

%% Create Output Tables
tffeatsRight = table(nomout,EpowR(:,1),EpowR(:,2),IpowR(:,1),IpowR(:,2),plfR(:,1),plfR(:,2));
tffeatsRight.Properties.VariableNames = {'contents','Evoked_Power_1','Evoked_Power_2','Induced_Power_1','Induced_Power_2','PLF_1','PLF_2'};
tffeatsLeft = table(nomout,EpowL(:,1),EpowL(:,2),IpowL(:,1),IpowL(:,2),plfL(:,1),plfL(:,2));
tffeatsLeft.Properties.VariableNames = {'contents','Evoked_Power_1','Evoked_Power_2','Induced_Power_1','Induced_Power_2','PLF_1','PLF_2'};

%% Save Data
filenamesave = [savepath,fname,'.mat'];
save(filenamesave,'timef_output','tffeatsRight','tffeatsLeft','-v7.3');

end
