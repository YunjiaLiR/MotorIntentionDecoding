%% load filename
clear,clc,close all
utildir = 'E:\Matlabcode\#6SuperMI4MA\utilities';
path(path,utildir);
folder = 'C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\';
for sub = 7
    data_folder = [folder,'Dataset\eeg\Subject ',num2str(sub),'\'];
    dirout = dir(fullfile(data_folder, '*.vhdr'));
    filenames = {dirout.name};
    for i = 1:length(filenames)
        EEGpreprocess(data_folder,filenames{i},['C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\Preprocess_data\eeg\step1_preprocess\Subject ',num2str(sub),'\']);
    end
end
%%
for sub = 2
    data_folder = ['C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\Preprocess_data\eeg\step1_preprocess\Subject ',num2str(sub),'\'];
    label_folder = ['C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\Dataset\force\Subject ',num2str(sub),'\'];
    dirout = dir(fullfile(data_folder, '*.set'));
    filenames = {dirout.name};
    Right = [];Rest = [];Left = [];
    for i = 1:length(filenames)
        dirout_label = dir(fullfile(label_folder, ['Run',num2str(i),'*']));
        filenames_label = {dirout_label.name};

        tokens = regexp(filenames_label, '^Run\d+Trial(\d+)\.mat$', 'tokens');
        trialNums = cellfun(@(t) str2double(t{1}), tokens);
        [~, order] = sort(trialNums);
        filenames_label_sorted = filenames_label(order);
        dirout_label = dirout_label(order);

        [left,right,rest] = cutEpoch1(filenames{i},data_folder,filenames_label_sorted,label_folder,5);
        Right = [Right right];
        Left = [Left left];
        Rest = [Rest rest];
    end
    save(['C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\Preprocess_data\sub',num2str(sub)], 'Right','Left','Rest');
end