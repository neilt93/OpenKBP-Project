# Prior AAPM poster text (style and language reference)

This is the full text of the group's prior AAPM poster, "Adversarial Robustness Evaluation of AI
Based Dose Prediction Models Using Clinical DVH Metrics" (Chowdhury, Tripathi, Ren, Sawant, Liao,
Vaishnav). The presenter wants the ASTRO poster written in the same voice, tone, and section
structure. The one substantive difference: the perturbations are now clinically realistic physical
models of CT variability (noise, calibration, bias field, resolution, dental artifact) rather than
adversarial attacks. The AAPM poster's future-work line explicitly announces this ASTRO study.

## ABSTRACT
Deep learning based dose prediction models are increasingly used in radiotherapy, yet their
robustness to input uncertainty remains poorly understood. We developed a framework to evaluate 3D
head-and-neck dose prediction models by applying controlled adversarial perturbations to CT images
and assessing changes in clinically relevant dose-volume metrics. Results demonstrated measurable
degradation in target coverage and organ-at-risk sparing, even for small perturbations, with
effects increasing systematically as perturbation magnitude increased. These findings highlight the
importance of incorporating robustness testing and uncertainty assessment into the development and
clinical evaluation of AI-based dose prediction models.

## INTRODUCTION
Deep learning based treatment planning models are increasingly being deployed clinically, yet their
evaluation is typically limited to prediction accuracy under standard test conditions.
Consequently, model robustness to input variability and uncertainty remains poorly understood.
Accuracy-based assessments may overlook vulnerabilities that can affect clinically relevant
outputs. To stress-test model performance, we apply FGSM and PGD adversarial perturbations to CT
inputs. These methods generate small, targeted input changes designed to maximize output
deviations, enabling systematic evaluation of robustness and model sensitivity.

## METHODS AND MATERIALS
Data set: Our framework is created using the Open Knowledge-Based Planning (OpenKBP) dataset, a
publicly available benchmark for radiotherapy dose prediction. 340 Head and Neck Patients.
Anonymized CT Images with PTV and OAR Contours. 70/63/56 Gy Prescriptions. Organs at Risk
contoured: Brainstem, Spinal Cord, Parotids, Larynx, Mandible, Esophagus.

Dose Prediction Model: 3D U-Net architecture. Data split: 200 training, 40 validation, 100 test
plans. Encoder-decoder network with skip connections. Inputs: CT image plus structure masks.
Output: voxel-wise 3D dose distribution. Trained using mean squared error loss between predicted
and clinical dose maps.

Adversarial perturbation model: Adversarial perturbations in test input data are optimized to
maximally impact model output.

## RESULTS
Measurable dosimetric changes were observed even at the smallest perturbation level (epsilon =
0.001), with decreases in PTV coverage metrics and increases in hotspot metrics. Increasing
perturbation magnitude produced progressive degradation in target coverage, with D95 reductions
reaching 6-9% at epsilon = 0.01 and exceeding 15% at epsilon = 0.05. OAR endpoints demonstrated
similar sensitivity, with notable increases in D0.1cc, V30, and V50, indicating reduced sparing
robustness at higher perturbation levels. Both FGSM and PGD attacks resulted in increasing
prediction error and performance degradation as epsilon increased, with PGD consistently producing
larger effects than FGSM. The greatest degradation occurred at the highest perturbation magnitudes,
demonstrating that deep learning dose prediction models are vulnerable to structured input
perturbations. Key Finding: Small CT intensity perturbations can lead to clinically meaningful
changes in predicted dose distributions despite strong baseline model performance.

## DISCUSSION
Deep learning dose prediction models demonstrated increasing sensitivity to CT perturbations as
adversarial magnitude increased. While overall prediction accuracy remained acceptable under
nominal conditions, clinically relevant DVH metrics showed measurable degradation even at small
perturbation levels. PGD attacks produced greater degradation than FGSM attacks, suggesting that
iterative perturbations more effectively exploit vulnerabilities in the learned dose mapping. These
findings indicate that conventional accuracy metrics alone may not adequately characterize model
reliability and that robustness testing can reveal important failure modes not observed during
standard validation.

## CONCLUSIONS
We adopted adversarial perturbation as a diagnostic tool for robustness of AI models in
radiotherapeutic dose prediction. From this study few key points emerge relevant to clinical
practice. Accurate AI models are not necessarily robust AI models. Small CT perturbations can
produce measurable dosimetric changes. Robustness testing can identify vulnerabilities not captured
by standard validation. Incorporating robustness evaluation may improve confidence in clinical
deployment of AI-based planning tools.

## FUTURE WORK
The noise models in this work may not represent actual physical noise, impact of which will be
presented in ASTRO 2026 as a poster presentation.

## REFERENCES
1. Babier A, Mahon R, McNiven A, Diamant A, Chan TCY. Med Phys. 2021;48:4932-4948.
2. Ronneberger O, Fischer P, Brox T. MICCAI. 2015;234-241.
3. Goodfellow IJ, Shlens J, Szegedy C. ICLR. 2015.

## CONTACT
Birjoo Vaishnav, PhD, DABR. University of Maryland School of Medicine, Baltimore MD.
bvaishnav@som.umaryland.edu. 410-427-2039 (o).
