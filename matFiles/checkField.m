function checkField(datasets_dir)
    target_vars = ["sa_adj", "te_adj", "pr_filt"];
    fid = fopen("bad_list.txt", "w");

    folders = dir(fullfile(datasets_dir, 'itp*cormat'));
    for i = 1:length(folders)
        folder_name = folders(i).name;
        folder_path = fullfile(datasets_dir, folder_name);
        mat_files = dir(fullfile(folder_path, '*.mat'));

        for j = 1:length(mat_files)
            file_name = mat_files(j).name;
            file_path = fullfile(folder_path, file_name);

            try
                info = h5info(file_path);
                vars = string({info.Datasets.Name});
                missing = setdiff(target_vars, vars);

                if ~isempty(missing)
                    fprintf(fid, "%s/%s | Missing: %s | Found: %s\n", ...
                        folder_name, file_name, strjoin(missing, ', '), strjoin(vars, ', '));
                end
            catch ME
                fprintf(fid, "%s/%s | Error: %s\n", folder_name, file_name, ME.message);
            end
        end
    end

    fclose(fid);
end
