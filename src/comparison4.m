clear; clc; close all;

% ======================================
%  SETUP PATHS
% =======================================
dataFolder = "C:\Users\lpz0003\Documents\COMP-6600---Final-Project\Data\";

sp3Files = dir(fullfile(dataFolder, "Lag1_*.sp3"));
tleFiles = dir(fullfile(dataFolder, "Lag1TLE_10125v2.txt"));

fprintf("Found %d SP3 files and %d TLE files.\n", ...
    length(sp3Files), length(tleFiles));

% ======================================
%  LOAD AND PARSE ALL TLE FILES
% =======================================

rawTLE = [];

for i = 1:length(tleFiles)
    tlePath = fullfile(dataFolder, tleFiles(i).name);
    fprintf("Parsing TLE file: %s\n", tleFiles(i).name);

    tleList = parseMultiTLE(tlePath);

    for k = 1:length(tleList)
        entry.line1 = tleList(k).line1;
        entry.line2 = tleList(k).line2;
        entry.filename = tleFiles(i).name;
        rawTLE = [rawTLE; entry];
    end
end

fprintf("\nParsed %d raw TLE pairs.\n", length(rawTLE));

% ======================================
%  CONVERT RAW TLE PAIRS TO MATLAB TLE OBJECTS
% =======================================

masterTLE = struct('line1',{},'line2',{},'filename',{}, ...
    'tleObj',{}, 'epoch',{});

for i = 1:length(rawTLE)

    tempPath = fullfile(tempdir, "tempTLE.txt");
    fid = fopen(tempPath, "w");
    fprintf(fid, "%s\n%s\n", rawTLE(i).line1, rawTLE(i).line2);
    fclose(fid);

    tleObj = tleread(tempPath);

    entry = rawTLE(i);
    entry.tleObj = tleObj;
    entry.epoch  = getTLEepoch(rawTLE(i).line1);

    masterTLE(end+1) = entry;
end

[~, idx] = sort([masterTLE.epoch]);
masterTLE = masterTLE(idx);

fprintf("TLE parsing complete. Earliest epoch = %s\n", ...
    datestr(masterTLE(1).epoch));


% ======================================================================
% ======================   LOOP OVER ALL SP3 FILES   ====================
% ======================================================================

for s = 1:length(sp3Files)

    fprintf("\n==========================================================\n");
    fprintf("Processing SP3 %d/%d: %s\n", s, length(sp3Files), sp3Files(s).name);
    fprintf("==========================================================\n");

    % ------------ Load SP3 file -----------------
    sp3Path = fullfile(dataFolder, sp3Files(s).name);
    sp3 = readSP3(sp3Path);

    sp3Times = datetime(sp3.epochs(:,1), sp3.epochs(:,2), sp3.epochs(:,3), ...
                        sp3.epochs(:,4), sp3.epochs(:,5), sp3.epochs(:,6), ...
                        "TimeZone","UTC");

    sp3Start = sp3Times(1);
    sp3End   = sp3Times(end);

    fprintf("SP3 Time Range: %s → %s\n", datestr(sp3Start), datestr(sp3End));

    % --------------------------------------------------------------
    % FIND TLE with epoch ≥ SP3 start
    % --------------------------------------------------------------

    idxTLE = find([masterTLE.epoch] >= sp3Start, 1, 'first');

    if isempty(idxTLE)
        warning("No TLE occurs after SP3 start — skipping.");
        continue;
    end

    chosenTLE = masterTLE(idxTLE);

    fprintf("Matched TLE: %s (epoch %s)\n", ...
        chosenTLE.filename, datestr(chosenTLE.epoch));

    % --------------------------------------------------------------
    % FILTER SP3 AFTER TLE EPOCH
    % --------------------------------------------------------------

    idx0 = find(sp3Times >= chosenTLE.epoch, 1, 'first');

    if isempty(idx0)
        warning("No usable SP3 samples after TLE epoch.");
        continue;
    end

    maskSP3 = idx0:length(sp3Times);

    utcTime = sp3Times(maskSP3);
    sp3Pos_ecef_use = sp3.position(maskSP3, :) * 1000;   % Convert km → m
    sp3Vel_ecef_use = sp3.velocity(maskSP3, :);          % Already m/s

    fprintf("Comparison window: %s → %s (%d samples)\n", ...
        datestr(utcTime(1)), datestr(utcTime(end)), numel(utcTime));

    % --------------------------------------------------------------
    % Convert SP3 → ECI
    % --------------------------------------------------------------

    N = numel(utcTime);
    sp3ECI = zeros(N,3);

    for i = 1:N
        sp3ECI(i,:) = ecef2eci(utcTime(i), sp3Pos_ecef_use(i,:), sp3Vel_ecef_use(i,:));
    end

    % --------------------------------------------------------------
    % SGP4 Propagation at 30-second resolution
    % --------------------------------------------------------------

    times = utcTime(1):seconds(30):utcTime(end);

    fprintf("Propagating SGP4...\n");
    [posSGP4, velSGP4] = propagateOrbit(times, chosenTLE.tleObj, ...
        "OutputCoordinateFrame","inertial");

    sgp4ECI_pos = posSGP4(:,:)';  % Nx3
    sgp4ECI_vel = velSGP4(:,:)';  % Nx3  <-- NEW

    % --------------------------------------------------------------
    % Interpolate SP3 truth to SGP4 time stamps
    % --------------------------------------------------------------

    Xint = interp1(utcTime, sp3ECI(:,1), times, 'spline');
    Yint = interp1(utcTime, sp3ECI(:,2), times, 'spline');
    Zint = interp1(utcTime, sp3ECI(:,3), times, 'spline');

    truthECI = [Xint', Yint', Zint'];

    % --------------------------------------------------------------
    % Compute Errors
    % --------------------------------------------------------------

    err = truthECI - sgp4ECI_pos;

    err_x = err(:,1);
    err_y = err(:,2);
    err_z = err(:,3);
    err_norm = vecnorm(err,2,2);

    % --------------------------------------------------------------
    % SAVE ERROR TABLE (with velocities)
    % --------------------------------------------------------------

    outName = erase(sp3Files(s).name, ".sp3") + "_errors.txt";
    outPath = fullfile(dataFolder, outName);

    T = table(times', ...
              sgp4ECI_pos(:,1), sgp4ECI_pos(:,2), sgp4ECI_pos(:,3), ...
              sgp4ECI_vel(:,1), sgp4ECI_vel(:,2), sgp4ECI_vel(:,3), ... % <-- NEW
              err_x, err_y, err_z, err_norm, ...
        'VariableNames', {'time', ...
                          'x_sgp4','y_sgp4','z_sgp4', ...
                          'vx_sgp4','vy_sgp4','vz_sgp4', ... % <-- NEW
                          'err_x','err_y','err_z','err_norm'});

    writetable(T, outPath, 'Delimiter','\t');

    fprintf("Saved error file: %s\n", outPath);

end

fprintf("\nAll SP3 segments processed.\n");
