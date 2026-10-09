clear; clc;
addpath('utilities')
subject = 1;
exp = 1;
day = 1;
runs = [1 2 3 4 11];
trainRuns = [1 2 3 4];
motorChannelIndices = [39 7 29 58 8 40 24 57 25 43 12 53 23 54];
motorChannels = ["FC3", "FC1", "FC2", "FC4", ...
    "C3", "C1", "Cz", "C2", "C4", ...
    "CP3", "CP1", "CPz", "CP2", "CP4"];
fs = 1000;

labelsToPlot = [5 6 7];
tfrChannel = 1;    
topoBand = [8 40];

[trainData, trainLabels, testData, testLabels] = prepare_data( ...
    subject, exp, day, runs, trainRuns, motorChannelIndices);

allData = [trainData; testData];
allLabels = [trainLabels; testLabels];

plot_tfr(allData, allLabels, labelsToPlot, fs, tfrChannel);
plot_topomap(allData, allLabels, labelsToPlot, fs, topoBand, motorChannels);