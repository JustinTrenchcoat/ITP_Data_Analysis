% pressure.m
function p = pressure(height, latitude)
    z = -height;
    p = gsw_p_from_z(z, latitude);
end