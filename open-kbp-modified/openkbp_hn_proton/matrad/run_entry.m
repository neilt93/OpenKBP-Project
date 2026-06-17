% Container entrypoint for the openkbp-matrad Octave image.
% Run with cwd = /opt/matRad (so matRad_rc is found) :
%   octave --no-gui /work/scripts/run_entry.m
% Mounts (see docker_octave.sh): /work/cases (<pid>_input.mat in),
% /work/out (<pid>_dose.mat out), /work/scripts (this dir, has run_plan.m).
matRad_rc;
% Headless: suppress the dose-calc waitbar (initDoseCalc guards it on this flag),
% otherwise Octave aborts with "no graphics toolkits are available!".
matRad_cfg = MatRad_Config.instance();
matRad_cfg.disableGUI = true;
addpath('/work/scripts');
run_plan('/work/cases', '/work/out');
