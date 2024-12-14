% fonction s
function y = s(t)
    n = 1;
    y = 0;
    while n < t
        y = y + (1/2)*n;
        n = n + 1;
    end
end