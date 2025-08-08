function process_gridDataMat()
    % MATLAB translation of your Python script
    % Requires: GSW Oceanographic Toolbox, helper.m (pressure, height functions)
    
    datasets_dir = 'gridDataMat';
    df_list = {}; % cell array to store tables
    
    folders = dir(datasets_dir);
    folders = folders([folders.isdir] & ~ismember({folders.name},{'.','..'}));
    
    for f = 1:length(folders)
        folder_name = folders(f).name;
        folder_path = fullfile(datasets_dir, folder_name);
        
        fprintf('\nProcessing folder: %s\n', folder_name);
        
        % Extract system number from folder name
        tokens = regexp(folder_name, '(\d+)', 'tokens');
        if ~isempty(tokens)
            system_num = str2double(tokens{1}{1});
        else
            system_num = NaN; % no number found
        end
        
        % Get all .mat files in folder
        mat_files = dir(fullfile(folder_path, '*.mat'));
        [~, idx] = sort({mat_files.name});
        mat_files = mat_files(idx);
        
        for m = 1:length(mat_files)
            file_name = mat_files(m).name;
            full_path = fullfile(folder_path, file_name);
            
            try
                % Extract profile number from file name
                tokens = regexp(file_name, '(\d+)', 'tokens');
                if ~isempty(tokens)
                    profile_num = str2double(tokens{1}{1});
                else
                    profile_num = NaN;
                end
                
                % Call singleRead equivalent
                df_list = singleRead(full_path, df_list, profile_num, system_num);
                
            catch ME
                fprintf('Error processing file: %s\n', file_name);
                disp(getReport(ME));
            end
        end
    end
    
    % Concatenate all tables
    final_df = vertcat(df_list{:});
    
    % Save as .mat (MATLAB equivalent to .pkl)
    save('test.mat', 'final_df');
end


function ls = singleRead(full_path, ls, profile_num, sys_num)
    % Load .mat data
    data = load(full_path);
    
    depth = data.Depth(:);
    lat = data.lat(:);
    lon = data.lon(:);
    psdate = string(data.startDate(:)');
    dateVal = datetime(psdate, 'InputFormat', 'MM/dd/yy');
    temp = data.Temperature(:);
    salinity = data.Salinity(:);
    pres = pressure(depth, lat); % from helper.m
    
    assert(~any(isnan(depth)), 'depth contains NaN');
    assert(~any(isnan(temp)), 'temp contains NaN');
    assert(~any(isnan(salinity)), 'salinity contains NaN');
    assert(~any(isnan(pres)), 'pres contains NaN');
    
    % Convert to Absolute Salinity and Conservative Temperature
    salinity = gsw_SA_from_SP(salinity, pres, lon, lat);
    temp = gsw_CT_from_t(salinity, temp, pres);
    
    % Create table
    T = table(depth, temp, repmat(dateVal(1), length(depth), 1), salinity, ...
              repmat(lon(1), length(depth), 1), repmat(lat(1), length(depth), 1), ...
              pres, 'VariableNames', ...
              {'depth','temp','date','salinity','lon','lat','pressure'});
    
    % Add gradients with Gaussian smoothing
    dTdz = gradient(temp, depth);
    dSdz = gradient(salinity, depth);
    T.("dT_dZ") = smoothdata(dTdz, 'gaussian', 80);
    T.("dS_dZ") = smoothdata(dSdz, 'gaussian', 80);
    
    % N^2 calculation
    [n_sq, ~] = gsw_Nsquared(salinity, temp, pres, lat(1));
    n_sq = smoothdata(n_sq, 'gaussian', 80);
    
    % Turner angle, R_rho
    [turner_angle, R_rho, p_mid] = gsw_Turner_Rsubrho(salinity, temp, pres);
    depth_mid = height(p_mid, lat(1)); % from helper.m
    R_rho = 1 ./ R_rho;
    R_rho(~(R_rho > 0 & R_rho < 100)) = 0;
    
    % Interpolation to depth grid
    interpolated_R_rho = interp1(depth_mid, R_rho, depth, 'linear', 'extrap');
    interpolated_turner = interp1(depth_mid, turner_angle, depth, 'linear', 'extrap');
    interpolated_n_sq = interp1(depth_mid, n_sq, depth, 'linear', 'extrap');
    
    % Smooth
    T.("n_sq") = smoothdata(interpolated_n_sq, 'gaussian', 80);
    T.("turner_angle") = smoothdata(interpolated_turner, 'gaussian', 80);
    T.("R_rho") = smoothdata(interpolated_R_rho, 'gaussian', 80);
    
    % Sanity check
    assert(height(T) == length(depth), 'Wrong dimension');
    
    % Add IDs
    T.("profileNum") = repmat(profile_num, height(T), 1);
    T.("systemNum") = repmat(sys_num, height(T), 1);
    
    % Append to cell list
    ls{end+1} = T;
end
