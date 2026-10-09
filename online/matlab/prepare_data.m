function [trainData, trainLabels, testData, testLabels,fs,windowLength,stepLength] = prepare_data(subject, exp, day, runs, trainRuns, motorChannelIndices)

addpath("utilities");
load("utilities\chanlocs64.mat");
folder = fullfile("Calibration_data",sprintf("Subject%d", subject),sprintf("exp%d", exp),sprintf("Day%d", day),"arm_oriented"); %,"arm_oriented"


%motorChannelIndices = [39 7 29 58 8 40 24 57 25 43 12 53 23 54];

trainData = {};
trainLabels = [];
testData = {};
testLabels = [];

for r = runs
    matFile = fullfile(folder, sprintf("sub%d_exp%d_run%d.mat", subject, exp, r));
    S = load(matFile);

    fs = S.parameter.samplingrate;
    windowLength = fs*S.parameter.windowLength;
    stepLength = round(S.parameter.predictEvery * fs);
    expectedWindowNum = S.parameter.subtrialnum;

    eeg = S.data;

    starts = S.event.start(:);
    labels = S.parameter.randomOrder(:);

    nTrials = length(labels);
    if length(starts) ~= nTrials
        error("Run %d: event starts and labels have different lengths.", r);
    end

    for t = 1:nTrials
        startSample = starts(t);
        trialLabel = labels(t);

        for w = 1:expectedWindowNum
            startWindow = startSample + (w - 1) * stepLength;
            endWindow = startWindow + windowLength - 1;

            if endWindow > size(eeg, 2)
                break
            end

            windowData = eeg(:, startWindow:endWindow);
            windowData = onlinePreProcess(windowData, fs, motorChannelIndices,chanlocs);

            if ismember(r, trainRuns)               
                trainData{end+1, 1} = windowData;
                trainLabels(end+1, 1) = trialLabel;
            else
                testData{end+1, 1} = windowData;
                testLabels(end+1, 1) = trialLabel;
            end
        end
    end
end

windowLength = S.parameter.windowLength;
stepLength = S.parameter.predictEvery;
fprintf("Training windows: %d\n", length(trainData));
fprintf("Testing windows: %d\n", length(testData));
end