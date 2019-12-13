function [envelopeEMG] = envelopeEMG(emgdata,emgfreq)
% envelopeEMG: a function to filter, rectify, and normalise emg data for
% use in muscle force analysis.
% NOTE: Based in part on code written by Ajay Seth for use with OpenSim,
% his work can be found here:
% http://simtk-confluence.stanford.edu:8080/display/OpenSim/Tools+for+Preparing+Motion+Data

% Use Butterworth filter (fourth-order) on emg data.
[b, a] = butter(2, [20 400]/(emgfreq/2), 'bandpass');
filteredEMG = filter(b, a, emgdata);

% Rectify band-pass data.
rectifiedEMG = abs(filteredEMG);

% Low-pass filter to create linear envelope.
[b, a] = butter(4, 10/(emgfreq/2));
lowpassEMG = filtfilt(b, a, rectifiedEMG);

% Normalise to 0-1 based on maximum and minimum values.
sortedEMG = sort(lowpassEMG);
zeroline = mean(sortedEMG(1:(length(lowpassEMG)/100),:));
minCutOff = ones(length(lowpassEMG),1)*zeroline;
maxCutOff = ones(length(lowpassEMG),1)*(max(lowpassEMG)-zeroline);
envelopeEMG = (lowpassEMG - minCutOff)./maxCutOff;
end