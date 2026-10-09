%% load filename
clear,clc,close all
utildir = 'E:\Matlabcode\#6SuperMI4MA\utilities';
path(path,utildir);

data_folder = 'C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\Preprocess_data\';
dirout = dir(fullfile(data_folder, '*.mat'));
filenames = {dirout.name};

%% cut epoch and calculate ERD ratio
load("chanlocs61.mat")
for i = 1:length(filenames)
    load([data_folder,filenames{i}]);
    [Ratio{i,1},AvgMag{i,1}] = EEGspectrum(8,30,Left);
    [Ratio{i,2},AvgMag{i,2}] = EEGspectrum(8,30,Right);
    [Ratio{i,3},AvgMag{i,3}] = EEGspectrum(8,30,Rest);
end

avgResult_Magleft = AverageMag(AvgMag_left);
avgResult_Magrest = AverageMag(AvgMag_rest);
avgResult_Ratioleft = AverageRatio(avgResult_Magleft);
avgResult_Ratiorest = AverageRatio(avgResult_Magrest);
%%
plotERD(Ratio,chanlocs)
% plot(avgResult_Ratiotask,'avgResult_Ratiotask',chanlocs)
%plot(avgResult_Ratiorest,'avgResult_Ratiorest',chanlocs)
%plot(Ratio_rest,'Ratio_rest',chanlocs)

%save([data_folder,'\exp\step2_2_epoch_power_analysis\ratio_exp.mat'],"Ratio_exp_t","Ratio_exp_r");
% filenameWithoutExt = strrep(call_list_E1{i,j}, '_1.mat', '');
% save([data_folder,'\exp\step2_1_epoch\',filenameWithoutExt,'.mat'],"TASK","REST");

function avg_results = AverageMag(AvgMag)
[num_people, num_states] = size(AvgMag);
avg_results = cell(1, num_states);
for state = 1:num_states
    current_state_data = AvgMag(:, state);
    combined_data = cat(4, current_state_data{:});
    avg_data = mean(combined_data, 4);
    avg_results{state} = avg_data;
end
end

function avgResult_Ratio = AverageRatio(avgResult)
for i = 1:size(avgResult,2)
    AvgMag = avgResult{i};
    for ch_select = 1:61 %channel number
        restf = mean(AvgMag(:,251:500,ch_select),2);%˝«1-2sĘýľÝ´ćČëtestA
        taskf = mean(AvgMag(:,502:1251,ch_select),2);%˝«2-5sĘýľÝ´ćČëtestB
        Ratio(1,ch_select) = 100*(sum(taskf(1:3,:))-sum(restf(1:3,:)))./sum(restf(1:3,:));%lalpha
        Ratio(2,ch_select) = 100*(sum(taskf(3:6,:))-sum(restf(3:6,:)))./sum(restf(3:6,:));%ualpha
        Ratio(3,ch_select) = 100*(sum(taskf(1:6,:))-sum(restf(1:6,:)))./sum(restf(1:6,:));%alpha
        Ratio(4,ch_select) = 100*(sum(taskf(7:13,:))-sum(restf(7:13,:)))./sum(restf(7:13,:));%lbeta
        Ratio(5,ch_select) = 100*(sum(taskf(14:23,:))-sum(restf(14:23,:)))./sum(restf(14:23,:));%ubeta
        Ratio(6,ch_select) = 100*(sum(taskf(7:23,:))-sum(restf(7:23,:)))./sum(restf(7:23,:));%beta
        Ratio(7,ch_select) = 100*(sum(taskf)-sum(restf))./sum(restf);%alphabeta
    end
    avgResult_Ratio{1,i} = Ratio;
end
end

function plotERD(ratio,chanlocs)
for side = 1:size(ratio,2)
    for sub = 1:size(ratio,1)
        plotratio = ratio{sub,side};
        for freband = 1:size(plotratio,1)
            figure; topoplot(plotratio(freband,:),chanlocs,'plotrad',0.6,'headrad',0.6, 'maplimits',[-20,20]);
            set(gcf, 'Renderer', 'painters');
            saveas(gcf,['C:\Users\Asus\OneDrive - Imperial College London\EEG4MI\Results\',['sub',num2str(sub),'_',num2str(side),'_',num2str(freband),'.tif']])
            close all
        end
    end
end
end