%{
This script goes through the rawData folder created by 01_pullData.py
and extracts data that satisfies the requirements:

1. Collected in the Beaufort Gyre
   (Start latitude between 73N and 81N, start longitude between 160W and 130W)
2. Measurement depth at least 10 m deeper than the AW Temperature Maximum
   AND 200 m below the surface

The new dataset will be stored in the goldData folder, organized by ITP number.

NaN values in salinity and other measurements (bad values) are filtered out.
Any observation with NaNs in any measurement is discarded.
%}

clear; clc;

% Paths
datasets_dir = 'rawData';
golden_dir = 'goldData';

% Keep track of bad profiles
bad_profile = {};

% Traverse all datasets
datasetFolders = dir(fullfile(datasets_dir, '*'));
datasetFolders = datasetFolders([datasetFolders.isdir]);
datasetFolders = datasetFolders(~ismember({datasetFolders.name}, {'.', '..'}));

for iFolder = 1:length(datasetFolders)
    folder_name = datasetFolders(iFolder).name;
    folder_path = fullfile(datasets_dir, folder_name);
    fprintf('Processing folder: %s\n', folder_name);

    matFiles = dir(fullfile(folder_path, '*.mat'));

    for iFile = 1:length(matFiles)
        filename = matFiles(iFile).name;
        full_path = fullfile(folder_path, filename);

        try
            data = load(full_path); % Assuming variables are stored directly

            % Read variables (adjust if stored differently)
            pr_filt = read_var(data, 'pr_filt');
            te_adj  = read_var(data, 'te_adj');
            sa_adj  = read_var(data, 'sa_adj');
            lat     = read_var(data, 'latitude');
            lon     = read_var(data, 'longitude');

            % Remove NaNs
            valid_mask = ~isnan(te_adj) & ~isnan(pr_filt) & ~isnan(sa_adj);
            pr_filt = pr_filt(valid_mask);
            te_adj  = te_adj(valid_mask);

            % Skip if empty
            if isempty(pr_filt) || isempty(te_adj)
                bad_profile{end+1} = filename; %#ok<AGROW>
                continue;
            end

            % Compute depth
            depth = height(pr_filt, lat);
            dep_max = max(depth);

            % Skip if not deep enough
            if dep_max <= 200
                bad_profile{end+1} = filename;
                continue;
            end

            % Criteria
            % 1. Beaufort Gyre
            is_Beaufort = (lat >= 73 && lat <= 81) && (lon >= -160 && lon <= -130);

            % 2. Temperature max below 200 m, but deeper than 250 m
            depth_index = find(depth >= 200);
            [~, temp_idx_max_rel] = max(te_adj(depth_index));
            temp_max_depth_idx = depth_index(temp_idx_max_rel);
            temp_max_depth = depth(temp_max_depth_idx);
            max_deep = (temp_max_depth >= 250);

            % 3. Deep enough: at least 10 m deeper than temp max
            deep_enough = (dep_max >= temp_max_depth + 10);

            % If passes all, copy file
            if deep_enough && is_Beaufort && max_deep
                dest_folder = fullfile(golden_dir, folder_name);
                if ~exist(dest_folder, 'dir')
                    mkdir(dest_folder);
                end
                copyfile(full_path, fullfile(dest_folder, filename));
            else
                bad_profile{end+1} = filename;
            end

        catch
            bad_profile{end+1} = filename;
        end
    end
end

fprintf('Total bad profiles: %d\n', numel(bad_profile));
