% This script downloads all level 3 high-resolution data
% in the .mat format into a new folder called rawData

% Base URL
baseUrl = "https://scienceweb.whoi.edu/itp/data/itpsys%d/itp%dcormat.zip";

% Create rawData directory
rawDataDir = "rawData";
if ~exist(rawDataDir, 'dir')
    mkdir(rawDataDir);
end

failList = {};

% Loop over ITP numbers
for itp_num = 1:143
    fprintf("Downloading ITP #%d...\n", itp_num);
    url = sprintf(baseUrl, itp_num, itp_num);
    zip_filename = sprintf("itp%dcormat.zip", itp_num);
    zip_path = fullfile(pwd, zip_filename);

    try
        % Download the file
        outfilename = websave(zip_path, url);
        fprintf("Downloaded: %s\n", zip_filename);

        % Extract to rawData/itp*cormat/
        extract_folder = fullfile(rawDataDir, sprintf("itp%dcormat", itp_num));
        if ~exist(extract_folder, 'dir')
            mkdir(extract_folder);
        end
        unzip(outfilename, extract_folder);
        fprintf("Extracted to: %s\n", extract_folder);

        % Delete the zip file
        delete(outfilename);

    catch ME
        fprintf("Failed to download or extract ITP #%d: %s\n", itp_num, ME.message);
        failList{end+1} = url; %#ok<SAGROW>
    end
end

% Print any failures
if ~isempty(failList)
    fprintf("Failed downloads:\n");
    disp(failList);
else
    fprintf("All downloads completed successfully.\n");
end
