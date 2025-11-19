function sp3Data = readSP3(fileName)
    % Open the file
    fid = fopen(fileName, 'r');
    if fid == -1
        error('Could not open the file.');
    end

    % Initialize variables
    sp3Data = struct();
    sp3Data.position = [];
    sp3Data.velocity = [];
    sp3Data.epochs = [];  % Initialize the epochs field

    % Read file line by line
    while ~feof(fid)
        line = fgetl(fid);

        % Check if the line starts with '*' (epoch time marker)
        if startsWith(line, '*')
            epoch = sscanf(line, '* %d %d %d %d %d %f'); % Read timestamp
            sp3Data.epochs(end + 1, :) = epoch'; % Store epoch time
        end

        % Check for position data (starts with 'PL51')
        if startsWith(line, 'PL51')
            pos = sscanf(line, 'PL51 %f %f %f'); % Extract position data
            sp3Data.position = [sp3Data.position; pos']; % Store in matrix
        end

        % Check for velocity data (starts with 'VL51')
        if startsWith(line, 'VL51')
            vel = sscanf(line, 'VL51 %f %f %f'); % Extract velocity data
            sp3Data.velocity = [sp3Data.velocity; vel']; % Store in matrix
        end
    end

    % Close file
    fclose(fid);
end