%% CFC IMAGES & STATS

setup_analysispath(1)
loadpath = ...
    '/data/rcho/MEG_NM_NR_testing/FINALMNE/ASSR_Paper/final_paper_items/resubmission2/wavelettest3/';
groups = {'controls','early','chronic'};
patterns = {'40cfc.mat','30cfc.mat','20cfc.mat'};
for ii = 1:numel(groups)
    a = dir([loadpath,groups{ii}]);
    a = {a.name}';
    eval(['contentsCFC.',groups{ii},'40 = a(~cellfun(@isempty,regexp(a,patterns{1})));']);
    eval(['contentsCFC.',groups{ii},'30 = a(~cellfun(@isempty,regexp(a,patterns{2})));']);
    eval(['contentsCFC.',groups{ii},'20 = a(~cellfun(@isempty,regexp(a,patterns{3})));']);
end

%% Mass Univariate Style Clustering
amps = 13:60;
phases = 4:12;
frqhood = zeros(length(amps),length(amps));
for iter = 1:length(amps)
    for ii = 1:length(amps)
        range = amps(iter)-1:amps(iter)+1;
        val = amps(ii);
        if ismember(val,range)
            frqhood(ii,iter) = 1;
        end
    end
end
frqhood = logical(frqhood);
[outputws] = loadmeg_comoddata2(contentsCFC,amps,phases,[40,30,20],1,loadpath,groups);

perms  = 10000;
[~, ~, clustinfR40, ~, ~]=clust_perm2(outputws.allgrps40srZ2,outputws.allgrps40brZ2,frqhood,perms,0.01,1,0.01,0,[],0);
[~, ~, clustinfL40, ~, ~]=clust_perm2(outputws.allgrps40slZ2,outputws.allgrps40blZ2,frqhood,perms,0.01,1,0.01,0,[],0);

% Raw stats for plotting
[~,~,~,STATS40r] = ttest2(output.allgrps40srZ2,output.allgrps40brZ2,'dim',3,'tail','right');
[~,~,~,STATS40l] = ttest2(output.allgrps40slZ2,output.allgrps40blZ2,'dim',3,'tail','right');

%% Plotting Variables

% Generate smoothed maps
[temp1,amps1,phases1] = smoothcomod(amps,phases,output.allgrps40brZ2);
[temp2,~,~] = smoothcomod(amps,phases,output.allgrps40bsZ2);
[temp3,~,~] = smoothcomod(amps,phases,output.allgrps40bsrZ2-output.allgrps40brZ2);
