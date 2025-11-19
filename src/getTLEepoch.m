function dt = getTLEepoch(line1)

    % Extract year
    yy = str2double(line1(19:20));
    if yy < 57
        year = 2000 + yy;
    else
        year = 1900 + yy;
    end

    % Extract fractional day-of-year
    dayFraction = str2double(line1(21:32));

    % Split into integer day + fractional part
    dayInt = floor(dayFraction);
    frac   = dayFraction - dayInt;

    % Convert fractional day into time
    secondsInDay = frac * 86400;

    dt0 = datetime(year,1,1,"TimeZone","UTC");
    dt  = dt0 + days(dayInt - 1) + seconds(secondsInDay);

end
