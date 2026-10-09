%% currently best
addpath('model\')
subject =1;
exp = 1;
day = 1;

runs = [1 2 3 4 11];
trainRuns = [1 2 3 4];
motorChannelIndice= [39 7 29 58 8 40 24 57 25 43 12 53 23 54];
frequencyBands = [8 12; 12 16; 16 20; 20 24; 24 28; 28 32; 32 36; 36 40];
nCspPairs = 1;


% Prepare data for training and testing
[trainData, trainLabels, testData, testLabels,fs,windowLength,predictEvery] = prepare_data(subject, exp, day, runs, trainRuns, motorChannelIndice);

% Train FBCSP model and extract training features
[featureModel, trainFeatures] = trainFBCSP(trainData, trainLabels, fs, frequencyBands, nCspPairs);
testFeatures = extract_fbcsp_features(featureModel, testData);

% Normalize features using training statistics only
[trainFeaturesZ, mu, sigma] = zscore(trainFeatures);
sigma(sigma == 0) = 1;
testFeaturesZ = (testFeatures - mu) ./ sigma;


% Feature Normalisation
scores = zeros(1, size(trainFeaturesZ, 2));

for j = 1:size(trainFeaturesZ, 2)
    scores(j) = abs(corr(trainFeaturesZ(:, j), double(trainLabels)));
end

[~, order] = sort(scores, "descend");

% Feature selection
nSelectedFeatures = 13;
selectedFeatures = order(1:nSelectedFeatures);

trainSelected = trainFeaturesZ(:, selectedFeatures);
testSelected = testFeaturesZ(:, selectedFeatures);

% Train classifier
classifier = train_classifier(trainSelected, trainLabels, "lda");

% Prediction
trainPred = predict(classifier, trainSelected);
predictedLabels = predict(classifier, testSelected);

trainAcc = mean(trainPred == trainLabels);
testAcc = mean(predictedLabels == testLabels);

fprintf("Selected train features: %d x %d\n", size(trainSelected,1), size(trainSelected,2));
fprintf("Selected test features:  %d x %d\n", size(testSelected,1), size(testSelected,2));
fprintf("Train accuracy: %.2f%%\n", trainAcc * 100);
fprintf("Test accuracy:  %.2f%%\n", testAcc * 100);

disp("Selected feature indices:")
disp(selectedFeatures)

confusionchart(testLabels, predictedLabels);

%% store model for online test
model = struct();
model.type = "online_fbcsp_lda";
model.fs = fs;
model.windowLength = windowLength;
model.predictEvery = predictEvery;
model.motorChannelIndices = motorChannelIndice;
model.frequencyBands = frequencyBands;
model.nCspPairs = nCspPairs;
model.featureModel = featureModel;
model.mu = mu;
model.sigma = sigma;
model.selectedFeatures = selectedFeatures;
model.classifier = classifier;

modelFile = fullfile('.', 'model', sprintf('OnlineFBCSPModel_Subject%d_exp%d_day%d.mat', subject, exp, day));

modelFolder = fileparts(modelFile);
if exist(modelFolder, 'dir') ~= 7
    mkdir(modelFolder);
end

save(modelFile, 'model');
fprintf("Saved online model to %s\n", modelFile);
