function z = height(pressure, latitude)
    z = -gsw_z_from_p(pressure, latitude);
end