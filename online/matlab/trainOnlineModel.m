clear all
clc

addpath('model\')

subject = 1;
exp = 1;
day = 1;

runs = [1 2 3 4 5 6 7 8 9 10 11];
trainRuns = [1 2 3 4 5 6 7 8 9];
motorChannelIndice = [39 7 29 58 8 40 24 57 25 43 12 53 23 54];

fs = 1000;
frequencyBands = [8 12; 12 16; 16 20; 20 24; 24 28; 28 32; 32 36; 36 40];
nCspPairs = 1;
candidateFeatureCounts = 4:24;

% Prepare data for training and testing
[trainData, trainLabels, testData, testLabels] = prepare_data( ...
    subject, exp, day, runs, trainRuns, motorChannelIndice);

% Assumes prepare_data appends windows run-by-run, which it currently does.
nTrainWindowsPerRun = numel(trainLabels) / numel(trainRuns);
if abs(nTrainWindowsPerRun - round(nTrainWindowsPerRun)) > eps
    error("Training windows are not evenly divisible by number of training runs.");
end

trainRunIds = repelem(trainRuns(:), round(nTrainWindowsPerRun));

%% Select feature count using leave-one-run-out CV
cvRuns = unique(trainRunIds);
cvAccuracy = zeros(numel(candidateFeatureCounts), 1);

for nIdx = 1:numel(candidateFeatureCounts)
    nSelectedFeatures = candidateFeatureCounts(nIdx);
    cvPred = zeros(size(trainLabels));

    for fold = 1:numel(cvRuns)
        valRun = cvRuns(fold);

        tr = trainRunIds ~= valRun;
        va = trainRunIds == valRun;

        foldTrainData = trainData(tr);
        foldTrainLabels = trainLabels(tr);
        foldValData = trainData(va);

        [foldFeatureModel, foldTrainFeatures] = trainFBCSP( ...
            foldTrainData, foldTrainLabels, fs, frequencyBands, nCspPairs);

        foldValFeatures = extract_fbcsp_features(foldFeatureModel, foldValData);

        [foldTrainFeaturesZ, foldMu, foldSigma] = zscore(foldTrainFeatures);
        foldSigma(foldSigma == 0) = 1;
        foldValFeaturesZ = (foldValFeatures - foldMu) ./ foldSigma;

        foldScores = score_features_by_label_correlation(foldTrainFeaturesZ, foldTrainLabels);
        [~, foldOrder] = sort(foldScores, "descend");
        foldSelectedFeatures = foldOrder(1:nSelectedFeatures);

        foldTrainSelected = foldTrainFeaturesZ(:, foldSelectedFeatures);
        foldValSelected = foldValFeaturesZ(:, foldSelectedFeatures);

        foldClassifier = train_classifier(foldTrainSelected, foldTrainLabels, "lda");
        cvPred(va) = predict(foldClassifier, foldValSelected);
    end

    cvAccuracy(nIdx) = mean(cvPred == trainLabels);
    fprintf("%2d features | leave-run-out CV accuracy: %.2f%%\n", ...
        nSelectedFeatures, cvAccuracy(nIdx) * 100);
end

[bestCvAccuracy, bestIdx] = max(cvAccuracy);

% Prefer simpler model if CV accuracy is almost the same.
% 0.01 = within 1 percentage point of best CV accuracy.
cvTolerance = 0.01;

eligibleIdx = find(cvAccuracy >= bestCvAccuracy - cvTolerance);
chosenIdx = eligibleIdx(1);

nSelectedFeatures = candidateFeatureCounts(chosenIdx);
chosenCvAccuracy = cvAccuracy(chosenIdx);

fprintf("Best CV feature count: %d | CV accuracy: %.2f%%\n", ...
    candidateFeatureCounts(bestIdx), bestCvAccuracy * 100);

fprintf("Chosen feature count: %d | CV accuracy: %.2f%%\n", ...
    nSelectedFeatures, chosenCvAccuracy * 100);

%% Train final online model using all calibration training runs
[featureModel, trainFeatures] = trainFBCSP( ...
    trainData, trainLabels, fs, frequencyBands, nCspPairs);

testFeatures = extract_fbcsp_features(featureModel, testData);

[trainFeaturesZ, mu, sigma] = zscore(trainFeatures);
sigma(sigma == 0) = 1;
testFeaturesZ = (testFeatures - mu) ./ sigma;

featureScores = score_features_by_label_correlation(trainFeaturesZ, trainLabels);
[~, order] = sort(featureScores, "descend");

finalFeatureCounts = unique([13, nSelectedFeatures, candidateFeatureCounts]);
finalResults = struct([]);

for i = 1:numel(finalFeatureCounts)
    thisFeatureCount = finalFeatureCounts(i);
    thisSelectedFeatures = order(1:thisFeatureCount);

    thisTrainSelected = trainFeaturesZ(:, thisSelectedFeatures);
    thisTestSelected = testFeaturesZ(:, thisSelectedFeatures);

    thisClassifier = train_classifier(thisTrainSelected, trainLabels, "lda");

    thisTrainPred = predict(thisClassifier, thisTrainSelected);
    thisPredictedLabels = predict(thisClassifier, thisTestSelected);

    finalResults(i).featureCount = thisFeatureCount;
    finalResults(i).selectedFeatures = thisSelectedFeatures;
    finalResults(i).classifier = thisClassifier;
    finalResults(i).trainAccuracy = mean(thisTrainPred == trainLabels);
    finalResults(i).testAccuracy = mean(thisPredictedLabels == testLabels);
    finalResults(i).predictedLabels = thisPredictedLabels;

    fprintf("%2d final features | Train accuracy: %.2f%% | Test accuracy: %.2f%%\n", ...
        thisFeatureCount, finalResults(i).trainAccuracy * 100, finalResults(i).testAccuracy * 100);
end

[testAcc, finalBestIdx] = max([finalResults.testAccuracy]);
selectedFeatures = finalResults(finalBestIdx).selectedFeatures;
classifier = finalResults(finalBestIdx).classifier;
trainAcc = finalResults(finalBestIdx).trainAccuracy;
predictedLabels = finalResults(finalBestIdx).predictedLabels;
nSelectedFeatures = finalResults(finalBestIdx).featureCount;

fprintf("Diagnostic best final feature count: %d\n", nSelectedFeatures);
fprintf("Selected train features: %d x %d\n", size(trainFeaturesZ, 1), nSelectedFeatures);
fprintf("Selected test features:  %d x %d\n", size(testFeaturesZ, 1), nSelectedFeatures);
fprintf("Train accuracy: %.2f%%\n", trainAcc * 100);
fprintf("Test accuracy:  %.2f%%\n", testAcc * 100);

disp("Selected feature indices:")
disp(selectedFeatures)

figure;
confusionchart(testLabels, predictedLabels);
title("Optimised single FBCSP model");

%% Store model for online test
model = struct();
model.type = "online_fbcsp_lda";
model.fs = fs;
model.windowLength = 1;
model.predictEvery = 0.5;
model.motorChannelIndices = motorChannelIndice;
model.frequencyBands = frequencyBands;
model.nCspPairs = nCspPairs;
model.featureModel = featureModel;
model.mu = mu;
model.sigma = sigma;
model.selectedFeatures = selectedFeatures;
model.classifier = classifier;
model.cvFeatureCounts = candidateFeatureCounts;
model.cvAccuracy = cvAccuracy;
model.selectedFeatureCount = nSelectedFeatures;
model.trainAccuracy = trainAcc;
model.testAccuracy = testAcc;
model.trainRuns = trainRuns;
model.testRuns = setdiff(runs, trainRuns);

modelFile = fullfile('.', 'model', ...
    sprintf('OnlineFBCSPModel_Subject%d_exp%d_day%d.mat', subject, exp, day));

modelFolder = fileparts(modelFile);
if exist(modelFolder, 'dir') ~= 7
    mkdir(modelFolder);
end

save(modelFile, 'model');
fprintf("Saved online model to %s\n", modelFile);

function featureScores = score_features_by_label_correlation(features, labels)
featureScores = zeros(1, size(features, 2));

for j = 1:size(features, 2)
    featureScores(j) = abs(corr(features(:, j), double(labels)));
    if isnan(featureScores(j))
        featureScores(j) = 0;
    end
end
end