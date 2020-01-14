function [smData,amp2,pha2] = smoothcomod(amps,phases,data)
% Nicholas Murphy (2020), Baylor College of Medicine, Houston, Texas, USA

if numel(size(data))==3
    [X,Y] = meshgrid(1:size(mean(data,3),2), 1:size(mean(data,3),1));
    [X2,Y2] = meshgrid(1:0.01:size(mean(data,3),2), 1:0.01:size(mean(data,3),1));
    smData = interp2(X, Y, mean(data,3), X2, Y2, 'spline');
    amp2 = amps(1):0.01:amps(end);
    pha2 = phases(1):0.01:phases(end);
else
    [X,Y] = meshgrid(1:size(data,2), 1:size(data,1));
    [X2,Y2] = meshgrid(1:0.01:size(data,2), 1:0.01:size(data,1));
    smData = interp2(squeeze(X), squeeze(Y), data, X2, Y2, 'spline');
    amp2 = amps(1):0.01:amps(end);
    pha2 = phases(1):0.01:phases(end);
end
end
