% instantiate the library
disp('Loading the library...');
lib = lsl_loadlib();

% resolve a stream...
disp('Resolving an EEG stream...');
result = {};
while isempty(result)
    result = lsl_resolve_byprop(lib,'type','EEG'); end

% create a new inlet
disp('Opening an inlet...');
inlet = lsl_inlet(result{1});

disp('Now receiving chunked data...');
stamps = cell(1);chunk = cell(1);i = 0;data=[];
[chunk{1},stamps{1}] = inlet.pull_chunk();
while true
    % get chunk from the inlet
    tic;
    pause(0.2);
    i = i+1;
    [chunk{i},stamps{i}] = inlet.pull_chunk();
    data = cat(2,data,chunk{i});
    event(i) = size(data,2);
    % for s=1:length(stamps)
    %     % and display it
    %     fprintf('%.2f\t',chunk(:,s));
    %     fprintf('%.5f\n',stamps(s));
    % end    
    toc
end