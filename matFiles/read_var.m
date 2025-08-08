function data = read_var(file_path, varname)
    try
        data = h5read(file_path, ['/' varname]);

        if isa(data, 'uint16')
            data = native2unicode(typecast(data(:), 'uint16'), 'UTF-16LE');
        end

        data = data(:); % flatten
    catch
        warning("Variable %s not found in %s", varname, file_path);
        data = nan;
    end
end
