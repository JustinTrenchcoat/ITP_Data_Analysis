function traverse_datasets(datasets_dir, func, errorFunc)
    folders = dir(datasets_dir);

    for i = 1:length(folders)
        folder_name = folders(i).name;
        folder_path = fullfile(datasets_dir, folder_name);

        if ~folders(i).isdir || ismember(folder_name, {'.', '..'})
            continue;
        end

        fprintf("\nProcessing folder: %s\n", folder_name);
        mat_files = dir(fullfile(folder_path, '*.mat'));

        for j = 1:length(mat_files)
            file_name = mat_files(j).name;
            full_path = fullfile(folder_path, file_name);

            try
                func(full_path, file_name, folder_name);
            catch
                if nargin > 2 && isa(errorFunc, 'function_handle')
                    errorFunc(file_name);
                end
            end
        end
    end
end
