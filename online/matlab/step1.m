%File read
clear,clc,close all
utildir = '.\utilities';
path(path,utildir);

subject = 1;%initial!!!
exp = 1;%initial!!!
expday = 1;

filefolder = '.\Calibration_data\Subject1\exp1\Day1\task_oriented';
dirout = dir(fullfile(filefolder, '*.mat'));
filenames = {dirout.name};
filenum = length(filenames);
load ('.\utilities\chanlocs64.mat') %!!!change channel location information
for i = 1:6
    eegfile = [filefolder,'\',filenames{i}];
    load(eegfile)
    
    [Data{trial},srate] = EEGpreprocess_liblsl(data,chanlocs);
end