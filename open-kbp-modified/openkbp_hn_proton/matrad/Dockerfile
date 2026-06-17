# matRad on GNU Octave 6.4.0 — for generating proton IMPT ground-truth dose.
#
# Ubuntu 22.04 (Jammy) ships Octave 6.4.0, which matches matRad's precompiled
# solver  thirdParty/IPOPT/ipopt.mexoct640a64  exactly. That mex is statically
# linked (IPOPT/MUMPS/METIS/BLAS/LAPACK baked in), so the only runtime deps are
# Octave itself + the gfortran runtime. No compilation needed.
#
# amd64 is REQUIRED: the mex is x86_64. On Apple Silicon, Docker runs it under
# emulation (slow but correct) — fine for the pt_201 gate; run the 240-patient
# batch on a native x86_64 Linux box (e.g. RunPod) with the same image.
#
# matRad is NOT baked in — it is mounted at runtime (see docker_octave.sh) so the
# image stays small and matRad can be updated independently.
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        octave \
        octave-statistics \
        gfortran \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && octave --version | head -1

WORKDIR /work
