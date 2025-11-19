function AllTLE = parseMultiTLE(tleFilePath)
% parseMultiTLE
% -----------------------
% Reads a TLE file containing multiple TLE sets.
% Each set contains:
%    AllTLE(k).line1  (char)
%    AllTLE(k).line2  (char)
%
% No datetime conversion.
% No parsing.
% Just raw strings.
%
% Works even if blank lines exist.

    % Read entire file and split into non-empty lines
    rawLines = strtrim(splitlines(fileread(tleFilePath)));
    rawLines = rawLines(~cellfun(@isempty, rawLines));

    AllTLE = struct('line1', {}, 'line2', {});

    idx = 1;

    % Step through file and collect TLE pairs
    for i = 1:length(rawLines)-1
        L1 = rawLines{i};
        L2 = rawLines{i+1};

        % Identify a valid TLE pair
        if startsWith(L1, '1 ') && startsWith(L2, '2 ')
            AllTLE(idx).line1 = char(L1);
            AllTLE(idx).line2 = char(L2);
            idx = idx + 1;
        end
    end
end
