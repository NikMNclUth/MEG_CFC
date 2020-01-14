function [output] = meg_feat_extract(contentsg,options,datanamer,datanamel,pathway,cond)
% extract wavelet features and thresholded vertices

% inputs
% contentsg = group contents cell
% options  = pre-processing options
% datanamer/l = right/left hem data name for loading purposes
% pathway = file path
% cond = entrainment frequency

% output
% this contains the left and right hemisphere vertices for each individual
% for 5 different methods. there are also timeseries and wavelet data
% generated for each of these so that data may be recreated.

% Nicholas Murphy (2020), Baylor College of Medicine, Houston, Texas, USA

for iter = 1:length(contentsg)
    %% load the data
    nom = contentsg{iter,1}(1:3);
    disp(nom)
    % load data
    a = [pathway,nom,datanamer]; % right hemisphere
    b = [pathway,nom,datanamel]; % left hemisphere
    load(a);
    load(b);
    %% reshape data to format = vertices, time, trials
    text1 = ['stcl=permute(stcs',num2str(cond),'_anat_bothL,[2,3,1]);'];
    text2 = ['stcr=permute(stcs',num2str(cond),'_anat_bothR,[2,3,1]);'];
    eval(text1);eval(text2);
    %% estimated best vertices and original transforms
    [tsL,wavoutL,vertsL] = thresholdROI(stcl,2,60,options.bl,options.wind,options.sr,cond,1);
    [tsR,wavoutR,vertsR] = thresholdROI(stcr,2,60,options.bl,options.wind,options.sr,cond,1);

    %% store data
    text1 = ['output.',nom,'.left.verts = vertsL;'];
    text2 = ['output.',nom,'.right.verts = vertsR;'];
    text3 = ['output.',nom,'.left.timeseriesL = tsL;'];
    text4 = ['output.',nom,'.right.timeseriesR = tsR;'];
    text5 = ['output.',nom,'.left.wavs = wavoutL;'];
    text6 = ['output.',nom,'.right.wavs = wavoutR;'];
    eval(text1);eval(text2);eval(text3);eval(text4);eval(text5);eval(text6);

    
end



end
