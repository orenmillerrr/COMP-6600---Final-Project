clear;clc;close all


sp3FileExt = ["Lag1_30225" "Lag1_30925" "Lag1_31625"];

position = [];
velocity = [];
epoch = [];
for i = 1 : length(sp3FileExt)

    fileName = char('D:\Classes\AI-orbit_estimation_agent\data\' + sp3FileExt(i) + '.sp3');
    
    data = readSP3(fileName);

    epoch = [epoch; data.epochs];
    position = [position; data.position * 10^3];
    velocity = [velocity; data.velocity * 0.1];

    idx = max(find(epoch(:,4) == 23));
    epoch = epoch(1:idx,:);
    position = position(1:idx,:);
    velocity = velocity(1:idx,:);
end

t0 = datetime(epoch(1,1),epoch(1,2),epoch(1,3),epoch(1,4),epoch(1,5),epoch(1,6),"TimeZone","UTC");
tf = datetime(epoch(end,1),epoch(end,2),epoch(end,3),epoch(end,4),epoch(end,5),epoch(end,6),"TimeZone","UTC");
dt = 60*2;
utcTime = t0 : seconds(dt) : tf;

k = 1;
for i = 1 : length(utcTime)
    [dataPos(k,:),dataVel(k,:)] = ecef2eci(utcTime(k),position(i,:),velocity(i,:));
    k = k + 1;
end

%% Read In TLE
tle = tleread("D:\Classes\AI-orbit_estimation_agent\data\Lag1TLE_30225.txt");

tle.Epoch.TimeZone = "UTC";
mask = utcTime >= tle.Epoch;
utcTime = utcTime(mask);
dataPos = dataPos(mask,:);
dataVel = dataVel(mask,:);


%% SGP4 Model
[SGP4pos,SGP4vel] = propagateOrbit(utcTime,tle,"PropModel","sgp4");
SGP4pos = SGP4pos';
SGP4vel = SGP4vel';

%% J2 Model
interpTime = t0 : seconds(1) : tf;
dataInterpPos(:,1) = interp1(utcTime,dataPos(:,1),interpTime,"spline");
dataInterpPos(:,2) = interp1(utcTime,dataPos(:,2),interpTime,"spline");
dataInterpPos(:,3) = interp1(utcTime,dataPos(:,3),interpTime,"spline");

J2pos = twobodyJ2(dataPos(1,:),dataVel(1,:),interpTime);
J2pos = J2pos';

%% Plots
figure
% plot3(SGP4pos(:,1),SGP4pos(:,2),SGP4pos(:,3))
hold on
plot3(J2pos(:,1),J2pos(:,2),J2pos(:,3))
plot3(dataPos(:,1),dataPos(:,2),dataPos(:,3),"--")

figure 
plot(dataPos(:,1) - SGP4pos(:,1))
hold on
plot(dataPos(:,2) - SGP4pos(:,2))
plot(dataPos(:,3) - SGP4pos(:,3))

figure 
plot(dataInterpPos(:,1) - J2pos(:,1))
hold on
plot(dataInterpPos(:,2) - J2pos(:,2))
plot(dataInterpPos(:,3) - J2pos(:,3))

% Norm SGP4 Error
figure
plot(utcTime,vecnorm(dataPos - SGP4pos,2,2))
grid on
xlabel("Time")
ylabel("Range Error (m)")

% Norm J2 Error
figure
plot(interpTime,vecnorm(dataInterpPos - J2pos,2,2))
grid on
xlabel("Time")
ylabel("Range Error (m)")


% function [ephem] = rnx2Ephem(rnx)
% 
%     [m,n] = size(rnx);
% 
%     for i = 1 : m
%         for j = 1 : n
%             ephem(i,j).prn       = rnx.SatelliteID(i);   
%             ephem(i,j).t_GD      = rnx.TGD(i); 
%             ephem(i,j).t_oc      = rnx.TransmissionTime(i);  
%             ephem(i,j).clkBias   = rnx.SVClockBias(i);
%             ephem(i,j).clkDrift  = rnx.SVClockDrift(i);
%             ephem(i,j).C_rc      = rnx.Crc(i);
%             ephem(i,j).C_rs      = rnx.Crs(i);
%             ephem(i,j).C_uc      = rnx.Cuc(i);
%             ephem(i,j).C_us      = rnx.Cus(i);
%             ephem(i,j).C_ic      = rnx.Cic(i);
%             ephem(i,j).C_is      = rnx.Cis(i);
%             ephem(i,j).Delta_n   = rnx.Delta_n(i);
%             ephem(i,j).M_0       = rnx.M0(i);
%             ephem(i,j).e         = rnx.Eccentricity(i);
%             ephem(i,j).sqrt_A    = rnx.sqrtA(i);
%             ephem(i,j).t_oe      = rnx.Toe(i);
%             ephem(i,j).Omega_0   = rnx.OMEGA0(i);
%             ephem(i,j).i_0       = rnx.i0(i);
%             ephem(i,j).omega     = rnx.omega(i);
%             ephem(i,j).dot_Omega = rnx.OMEGA_DOT(i);
%             ephem(i,j).Idot      = rnx.IDOT(i);  
%             ephem(i,j).a0      = rnx.SVClockBias(i);  
%             ephem(i,j).a1      = rnx.SVClockDrift(i);  
%             ephem(i,j).a2      = rnx.SVClockDriftRate(i);  
%             break
%         end
%     end
% end
% 
% 
