function countData(datasets_dir)
    total_profiles = 0;
    total_itps = 0;

    folders = dir(fullfile(datasets_dir, 'itp*cormat'));
    for i = 1:length(folders)
        folder_name = folders(i).name;
        folder_path = fullfile(datasets_dir, folder_name);
        mat_files = dir(fullfile(folder_path, '*.mat'));
        profile_count = length(mat_files);

        if profile_count == 0
            fprintf("Deleted empty folder: %s\n", folder_name);
        else
            total_profiles = total_profiles + profile_count;
            fprintf("%s: %d profiles\n", folder_name, profile_count);
        end

        total_itps = total_itps + 1;
    end

    fprintf("\nTotal number of remaining profiles: %d\n", total_profiles);
    fprintf("Total number of ITP systems: %d\n", total_itps);
end
