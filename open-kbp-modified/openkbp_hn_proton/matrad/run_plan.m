function run_plan(inputDir, outputDir)
% RUN_PLAN  Generate proton IMPT ground-truth dose for OpenKBP H&N cases.
%
%   run_plan(inputDir, outputDir)
%
%   Reads every <pid>_input.mat produced by build_case.py, builds a matRad
%   ct/cst/pln with a FIXED beam template + objectives, optimizes an IMPT plan,
%   and writes <pid>_dose.mat containing RBExDose and physicalDose cubes.
%
%   Written against matRad master (Octave 6.4.0). The plan flow mirrors
%   examples/matRad_example5_protons.m of that version exactly:
%     pln.bioModel='constRBE'; pln.multScen='nomScen'; engine='HongPB';
%     quantityOpt='RBExDose'; dij=matRad_calcDoseInfluence(ct,cst,stf,pln).
%   Run inside the openkbp-matrad Octave container (see docker_octave.sh), which
%   puts matRad on the path via matRad_rc before this is called.
%
%   Geometry: input cubes are 128x128x128 with logical indexing identical to the
%   OpenKBP cube (scipy.io preserved it across the C/F order difference). matRad
%   computes find(mask) itself, so there is NO index-convention mismatch. We keep
%   doseGrid resolution == ct resolution so RBExDose comes back on the same 128^3
%   grid, directly importable by import_dose.py.

    if nargin < 1 || isempty(inputDir)
        inputDir = fullfile(fileparts(mfilename('fullpath')), '..', 'matrad_cases');
    end
    if nargin < 2 || isempty(outputDir)
        outputDir = inputDir;
    end
    if ~exist(outputDir, 'dir'); mkdir(outputDir); end

    files = dir(fullfile(inputDir, '*_input.mat'));
    if isempty(files)
        error('run_plan:noInput', 'No *_input.mat in %s. Run build_case.py first.', inputDir);
    end

    for f = 1:numel(files)
        inPath  = fullfile(inputDir, files(f).name);
        pid     = strrep(files(f).name, '_input.mat', '');
        outPath = fullfile(outputDir, [pid '_dose.mat']);
        if exist(outPath, 'file')
            fprintf('Skipping %s (dose already exists)\n', pid);
            continue;
        end
        fprintf('\n==================  %s  ==================\n', pid);
        try
            plan_one(inPath, outPath);
        catch ME
            % getReport is unimplemented in Octave; print message + stack portably.
            fprintf(2, 'FAILED %s: %s\n', pid, ME.message);
            for s = 1:numel(ME.stack)
                fprintf(2, '    at %s (line %d)\n', ME.stack(s).name, ME.stack(s).line);
            end
        end
    end
    fprintf('\nDone. Import with: python openkbp_hn_proton/import_dose.py --result <pid>_dose.mat --out ...\n');
end


function plan_one(inPath, outPath)
    S = load(inPath);

    % ---- ct -----------------------------------------------------------------
    % Saved cube dims: dim1=axis0=A-P, dim2=axis1=L-R, dim3=axis2=S-I (resolved
    % on real data). matRad convention: resolution.x<->dim2, .y<->dim1, .z<->dim3.
    % S.resolution = [axis0, axis1, axis2] mm = [A-P, L-R, S-I].
    ct.cubeDim       = double(S.cubeDim(:)');
    ct.resolution.y  = double(S.resolution(1));   % dim1 = A-P
    ct.resolution.x  = double(S.resolution(2));   % dim2 = L-R
    ct.resolution.z  = double(S.resolution(3));   % dim3 = S-I
    ct.numOfCtScen   = 1;
    ct.cubeHU{1}     = double(S.cubeHU);
    % world axes (some matRad helpers expect them)
    ct.x = ((1:ct.cubeDim(2)) - 1) * ct.resolution.x;
    ct.y = ((1:ct.cubeDim(1)) - 1) * ct.resolution.y;
    ct.z = ((1:ct.cubeDim(3)) - 1) * ct.resolution.z;

    % ---- cst ----------------------------------------------------------------
    names = fieldnames(S.masks);
    cst = cell(numel(names), 6);
    for i = 1:numel(names)
        nm   = names{i};
        mask = logical(S.masks.(nm));
        typ  = S.structureType.(nm);

        cst{i,1} = i - 1;
        cst{i,2} = nm;
        cst{i,3} = typ;                 % 'TARGET' or 'OAR'
        cst{i,4} = {uint32(find(mask))};

        cst{i,5}.TissueClass  = 1;
        cst{i,5}.alphaX       = 0.1;
        cst{i,5}.betaX        = 0.05;
        cst{i,5}.Priority     = strcmp(typ, 'OAR') + 1;   % targets higher priority (lower #)
        cst{i,5}.Visible      = 1;
        cst{i,5}.visibleColor = [0.5 0.5 0.5];

        if strcmp(typ, 'TARGET')
            presc = double(S.prescription.(nm));
            % SquaredDeviation keeps dose AT prescription; an extra
            % SquaredUnderdosing term doubles the gradient against cold spots so
            % the optimizer drives up under-covered target voxels.
            cst{i,6}{1} = DoseObjectives.matRad_SquaredDeviation(double(S.targetPenalty), presc);
            cst{i,6}{2} = DoseObjectives.matRad_SquaredUnderdosing(double(S.targetPenalty), presc);
        else
            lim = double(S.oarMaxDose.(nm));
            cst{i,6}{1} = DoseObjectives.matRad_SquaredOverdosing(double(S.oarPenalty), lim);
        end
    end

    % ---- pln (FIXED protocol; mirrors example5_protons of this matRad) -------
    pln.radiationMode = S.radiationMode;     % 'protons'
    pln.machine       = S.machine;           % 'Generic'
    pln.bioModel      = 'constRBE';          % -> resultGUI.RBExDose (RBE 1.1)
    pln.multScen      = 'nomScen';           % nominal scenario (v1)
    pln.numOfFractions = 35;                 % ~2 Gy(RBE)/fx to 70

    pln.propStf.bixelWidth   = double(S.bixelWidth);
    pln.propStf.gantryAngles = double(S.gantryAngles(:)');
    pln.propStf.couchAngles  = double(S.couchAngles(:)');
    pln.propStf.numOfBeams   = numel(pln.propStf.gantryAngles);
    pln.propStf.isoCenter    = matRad_getIsoCenter(cst, ct, 0);

    pln.propOpt.runDAO        = 0;
    pln.propSeq.runSequencing = 0;
    pln.propOpt.quantityOpt   = 'RBExDose';

    % matRad's IPOPT default stops at "acceptable level" once the objective
    % change is < 1e-4 for 5 iters, which quits on a slow-progress plateau and
    % leaves cold target holes (D95=0) + OAR hotspots. Tighten so it converges;
    % options struct merges recursively, keeping the other IPOPT defaults.
    pln.propOpt.optimizer = struct('name', 'IPOPT');
    pln.propOpt.optimizer.options.acceptable_obj_change_tol = 1e-7;
    pln.propOpt.optimizer.options.max_iter                  = 1000;

    pln.propDoseCalc.calcLET = 0;
    pln.propDoseCalc.engine  = 'HongPB';     % analytical pencil beam
    % Compute dose on a FINE grid (proton penumbra/peak is ~3-5 mm; the OpenKBP CT
    % is a coarse 5.422 mm laterally). matRad_calcCubes then trilinearly interpolates
    % the result onto the 128^3 CT grid (calcCubes.m:224); a dose grid == the coarse
    % CT grid skips that and leaves zero-dose speckle holes -> D95=0.
    % 4 mm (not 3 mm) keeps dij memory ~constant when paired with 3 mm spots so the
    % whole thing fits the 7.6 GB local container. On a bigger box, drop to 3 mm.
    pln.propDoseCalc.doseGrid.resolution.x = 4;
    pln.propDoseCalc.doseGrid.resolution.y = 4;
    pln.propDoseCalc.doseGrid.resolution.z = 4;

    % ---- generate, calc, optimize ------------------------------------------
    stf       = matRad_generateStf(ct, cst, pln);
    dij       = matRad_calcDoseInfluence(ct, cst, stf, pln);
    resultGUI = matRad_fluenceOptimization(dij, cst, pln);

    % ---- extract dose cubes -------------------------------------------------
    % matRad_fluenceOptimization divides objective prescriptions by
    % numOfFractions and optimizes PER-FRACTION dose, so resultGUI dose is
    % per-fraction (~2 Gy/fx). OpenKBP photon dose is TOTAL-course dose (PTV70
    % ~70 Gy), so scale up by numOfFractions to match and keep the /70 training
    % normalization consistent.
    nFrac = double(pln.numOfFractions);
    if isfield(resultGUI, 'RBExDose')
        RBExDose = resultGUI.RBExDose * nFrac;
    else
        RBExDose = resultGUI.physicalDose * double(S.rbe) * nFrac;
    end
    if isfield(resultGUI, 'physicalDose')
        physicalDose = resultGUI.physicalDose * nFrac;
    else
        physicalDose = RBExDose / double(S.rbe);
    end

    % ---- plan QC (coverage / OAR sparing) -----------------------------------
    print_qc(S, RBExDose);

    save(outPath, 'RBExDose', 'physicalDose', '-v7');
    fprintf('  wrote %s\n', outPath);
end


function print_qc(S, dose)
    fprintf('  --- plan QC (Gy(RBE)) ---\n');
    tnames = {'PTV70','PTV63','PTV56'};
    for k = 1:numel(tnames)
        if isfield(S.masks, tnames{k})
            d = dose(logical(S.masks.(tnames{k})));
            fprintf('  %-10s D95 = %5.1f  Dmean = %5.1f  (presc %g)\n', ...
                tnames{k}, pctl(d,5), mean(d), S.prescription.(tnames{k}));
        end
    end
    onames = {'LeftParotid','RightParotid','SpinalCord','Brainstem'};
    for k = 1:numel(onames)
        if isfield(S.masks, onames{k})
            d = dose(logical(S.masks.(onames{k})));
            fprintf('  %-12s Dmean = %5.1f  Dmax = %5.1f  (limit %g)\n', ...
                onames{k}, mean(d), max(d), S.oarMaxDose.(onames{k}));
        end
    end
    fprintf('  EXPECT: PTV70 D95 ~>= 95%% of 70 for a usable plan.\n');
end


function v = pctl(x, p)
    % percentile without Statistics toolbox (works in MATLAB and Octave)
    x = sort(x(:));
    if isempty(x); v = NaN; return; end
    idx = max(1, min(numel(x), round(p/100 * numel(x))));
    v = x(idx);
end
