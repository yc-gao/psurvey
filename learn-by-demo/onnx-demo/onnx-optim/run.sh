#!/bin/sh

./main.py \
    --optimizer eliminate-identity \
    --optimizer convert-constant-to-initializer \
    --optimizer fold-constant \
    --optimizer convert-shape-to-initializer \
    --optimizer fold-constant \
    --optimizer eliminate-reshape \
    --optimizer eliminate-cast \
    --optimizer eliminate-concat \
    -o demo.onnx qualcomm_det2d_lane3d_det3d_occ_mono.onnx
