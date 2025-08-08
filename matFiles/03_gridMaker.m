function makeMatGrid(full_path, file_name, folder_name)
    persistent mat_error_list
    if isempty(mat_error_list)
        mat_error_list = {};
    end

    try
        % Read variables from HDF5 file
        pr_filt = read_var(full_path, 'pr_filt');
        lat = read_var(full_path, 'latitude');
        lon = read_var(full_path, 'longitude');
        psdate = read_var(full_path, 'psdate');
        pstart = read_var(full_path, 'pstart');
        pedate = read_var(full_path, 'pedate');
        pstop = read_var(full_path, 'pstop');
        te_adj = read_var(full_path, 'te_adj');
        sa_adj = read_var(full_path, 'sa_adj');

        % Filter out NaNs
        valid_mask = ~isnan(te_adj) & ~isnan(pr_filt) & ~isnan(sa_adj);
        pr_filt = pr_filt(valid_mask);
        te_adj = te_adj(valid_mask);
        sa_adj = sa_adj(valid_mask);

        % Step 1: Sort by depth
        depth = height(pr_filt, lat);
        [depths_sorted, sort_idx] = sort(depth);
        temperatures_sorted = te_adj(sort_idx);
        salinity_sorted = sa_adj(sort_idx);

        % Check for abnormal depth measurements
        if max(depth) > 800
            mat_error_list{end+1} = sprintf('%s: abnormal depth', full_path);
            error('Abnormal Depth of %f! Please check profile %s', max(depth), full_path);
        end

        % Check for empty arrays
        if isempty(depths_sorted) || isempty(temperatures_sorted) || isempty(salinity_sorted)
            mat_error_list{end+1} = sprintf('%s: empty entries', full_path);
            error('File %s has empty entries', full_path);
        end

        % Find T_Max first (temperature max below 200m)
        deep_index = find(depths_sorted >= 200);
        [~, temp_max_rel_idx] = max(temperatures_sorted(deep_index));
        temp_max_depth_idx = deep_index(temp_max_rel_idx);
        T_Max_Depth = depths_sorted(temp_max_depth_idx);

        % Find T_min between 120m and T_Max_Depth
        surface_index = find(depths_sorted >= 120 & depths_sorted <= T_Max_Depth);
        [~, temp_min_rel_idx] = min(temperatures_sorted(surface_index));
        temp_min_depth_idx = surface_index(temp_min_rel_idx);

        filter_range = temp_min_depth_idx : min(temp_max_depth_idx + 8, length(depths_sorted));
        depth_filtered = depths_sorted(filter_range);
        temp_filtered = temperatures_sorted(filter_range);
        sal_filtered = salinity_sorted(filter_range);

        % Step 2: Create regular depth grid (0.25 m spacing)
        start_depth = floor(min(depth_filtered));
        end_depth = ceil(max(depth_filtered));
        regular_depths = (start_depth:0.25:end_depth)';

        % Check for too few observations
        if length(regular_depths) < 18
            mat_error_list{end+1} = sprintf('%s: lack enough points', full_path);
            error('%s does not have enough valid points.', full_path);
        end

        % Check for duplicate depth values in filtered data
        [unique_vals, ~, ic] = unique(depth_filtered);
        counts = accumarray(ic, 1);
        duplicates = unique_vals(counts > 1);
        if ~isempty(duplicates)
            mat_error_list{end+1} = sprintf('%s: replicate value', full_path);
            warning('Duplicates found for file %s', full_path);
        end

        % Step 3: Interpolate temperature and salinity on regular grid
        interpolated_temperatures = interp1(depth_filtered, temp_filtered, regular_depths, 'linear', 'extrap');
        interpolated_salinity = interp1(depth_filtered, sal_filtered, regular_depths, 'linear', 'extrap');

        % Define physical ranges
        TEMP_MIN = -4; TEMP_MAX = 4;
        SAL_MIN = 0; SAL_MAX = 42;

        % Check for NaNs
        if any(isnan(interpolated_temperatures))
            nan_indices = find(isnan(interpolated_temperatures));
            mat_error_list{end+1} = sprintf('%s: NaNs found in interpolated temperatures at indices %s', full_path, mat2str(nan_indices));
            error('NaNs found in interpolated temperatures for file %s', full_path);
        end

        if any(isnan(interpolated_salinity))
            nan_indices = find(isnan(interpolated_salinity));
            mat_error_list{end+1} = sprintf('%s: NaNs found in interpolated salinity at indices %s', full_path, mat2str(nan_indices));
            error('NaNs found in interpolated salinity for file %s', full_path);
        end

        % Check for out-of-range values
        temp_out_of_range_idx = find(interpolated_temperatures < TEMP_MIN | interpolated_temperatures > TEMP_MAX);
        if ~isempty(temp_out_of_range_idx)
            mat_error_list{end+1} = sprintf('%s: interpolated temperatures out of range at indices %s', full_path, mat2str(temp_out_of_range_idx));
            error('Interpolated temperatures out of physical range in file %s', full_path);
        end

        sal_out_of_range_idx = find(interpolated_salinity < SAL_MIN | interpolated_salinity > SAL_MAX);
        if ~isempty(sal_out_of_range_idx)
            mat_error_list{end+1} = sprintf('%s: interpolated salinity out of range at indices %s', full_path, mat2str(sal_out_of_range_idx));
            error('Interpolated salinity out of physical range in file %s', full_path);
        end

        % Verify lengths match
        if ~(length(regular_depths) == length(interpolated_temperatures) && length(interpolated_temperatures) == length(interpolated_salinity))
            mat_error_list{end+1} = sprintf('%s: mismatch length', full_path);
            error('Length mismatch in interpolated arrays in file %s', full_path);
        end

        % Prepare output structure
        mdic = struct();
        mdic.Depth = regular_depths;
        mdic.Temperature = interpolated_temperatures;
        mdic.Salinity = interpolated_salinity;
        mdic.lat = lat;
        mdic.lon = lon;
        mdic.startDate = psdate;
        mdic.startTime = pstart;
        mdic.endDate = pedate;
        mdic.endTime = pstop;

        % Output path
        output_subfolder = fullfile('gridDataMat', folder_name);
        if ~exist(output_subfolder, 'dir')
            mkdir(output_subfolder);
        end
        output_path = fullfile(output_subfolder, file_name);

        % Save .mat file
        save(output_path, '-struct', 'mdic');

    catch ME
        % Log error message and rethrow or handle
        mat_error_list{end+1} = sprintf('%s: %s', full_path, ME.message);
        rethrow(ME);
    end
end
