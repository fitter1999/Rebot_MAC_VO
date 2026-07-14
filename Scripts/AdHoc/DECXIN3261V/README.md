# DECXIN-3261V MAC-VO Workflow

This folder contains the DECXIN-3261V specific live VO, calibration, probing, and map export entry points.

The DECXIN-3261V is treated as one UVC stream containing a horizontal stereo pair:

- Raw stream: `4000x1200`
- Physical left crop: `x=2080..3999`
- Physical right crop: `x=160..2079`
- Physical stereo baseline: `60 mm`
- Default display/capture image: `640x480`
- Default VO input: `640x480`
- The 1920x1200 eye image is center-cropped to the target aspect ratio before resizing. For the default `640x480` mode this is a 4:3 center crop, then resize.

## 1. Check Camera Node

```bash
lsusb
ls -l /dev/video* /dev/v4l/by-id /dev/v4l/by-path
gst-device-monitor-1.0 Video/Source
```

The camera should appear as `1bcf:2d50 Sunplus Innovation Technology Inc. DECXIN Camera` and should also have a `/dev/videoN` or `/dev/v4l/by-id/*DECXIN*video-index0` node.

If `lsusb` sees it but `/dev/videoN` is missing, unplug/replug the camera or move it to another USB3 port before running MAC-VO.

## 2. Probe Split Images

```bash
./run_macvo_wjy.sh python Scripts/AdHoc/DECXIN3261V/Probe.py --try-opencv
```

Probe images are written under `DECXIN-3261V-message/probe`.

## 3. Capture Calibration Images

```bash
Scripts/AdHoc/DECXIN3261V/capture_calibration.sh
```

This saves stereo checkerboard pairs to `Calibration/decxin3261v_screen_640x480`. The default checkerboard square size is `28 mm`.

## 4. Calibrate

```bash
Scripts/AdHoc/DECXIN3261V/calibrate.sh
```

The calibration result should include:

```text
Calibration/decxin3261v_screen_640x480/calibration_result/calibration.yaml
Calibration/decxin3261v_screen_640x480/calibration_result/rectify_maps.npz
```

## 5. Live VO

After calibration, run the realtime mode:

```bash
./run_decxin3261v_live_wjy.sh \
  --rectify-maps Calibration/decxin3261v_screen_640x480/calibration_result/rectify_maps.npz \
  --useRR
```

For slower quality comparison using parameters close to `MACVO_Fast.yaml` / `MACVO_Performant.yaml`:

```bash
./run_decxin3261v_quality_wjy.sh --useRR
```

For a paper-reproduce-style comparison using the main modules from `Paper_Reproduce.yaml`:

```bash
./run_decxin3261v_paperlike_wjy.sh --useRR
```

For MAC-VO mapping output closer to the project demo point-cloud visualization:

```bash
./run_decxin3261v_mapping_wjy.sh --useRR
```

This enables `Odometry.args.mapping: true`, which stores additional `map_points` selected from low-uncertainty stereo depth. It is a dense-mapping-oriented MAC-VO point cloud, but not a fused TSDF/mesh reconstruction.

To save frame images for later Rerun replay, record the sequence while mapping:

```bash
./run_decxin3261v_mapping_wjy.sh \
  --useRR \
  --record-sequence
```

For battery-powered outdoor capture, record only rectified stereo images at 10 Hz and show left/right images in Rerun. This does not run MAC-VO, so it does not require CUDA and does not show a live trajectory:

```bash
./run_decxin3261v_live_wjy.sh \
  --capture-only \
  --record-sequence \
  --useRR \
  --vo-fps 10 \
  --rr-every 1 \
  --status-every 30
```

Stop with `Ctrl+C`, then run offline mapping from the saved capture folder. For large rotations or close indoor objects, start with the paper-like mapping config. It follows the original `Paper_Reproduce.yaml` module choices, with TartanMotionNet initialization, ICP optimization, and outlier filters, while enabling mapping:

```bash
Scripts/AdHoc/DECXIN3261V/offline_map_from_sequence.sh \
  --result Results_decxin3261v_live/<time_dir> \
  --target-fps 3 \
  --odom Config/Experiment/MACVO/MACVO_DECXIN3261V_PaperLike_Mapping.yaml \
  --timing
```

For a faster comparison, use the default mapping config. It is closer to `MACVO_Fast.yaml` and uses StaticMotionModel plus the disp graph, but it can drift more on large rotations:

```bash
Scripts/AdHoc/DECXIN3261V/offline_map_from_sequence.sh \
  --result Results_decxin3261v_live/<time_dir> \
  --target-fps 3 \
  --timing
```

After the 3 Hz result is stable, full-frame offline mapping can be tested for comparison. Full-frame mode feeds every 10/30 Hz captured frame to MAC-VO, which is not equivalent to realtime `--vo-fps 3`:

```bash
Scripts/AdHoc/DECXIN3261V/offline_map_from_sequence.sh \
  --result Results_decxin3261v_live/<time_dir> \
  --odom Config/Experiment/MACVO/MACVO_DECXIN3261V_PaperLike_Mapping.yaml \
  --timing
```

Offline mapping defaults to `Config/Experiment/MACVO/MACVO_DECXIN3261V_Mapping.yaml`, not `MACVO_DECXIN3261V_Quality.yaml`. Both use the same `MACVO_FrontendCov.pth` frontend model and similar quality-oriented parameters, but the Mapping config enables `mapping: true` and `mapping_num_point: 2000`; the Quality config has `mapping: false`.

## 6. Reopen Saved Frame-by-frame Map in Rerun

Open the latest saved mapping result:

```bash
./run_decxin3261v_view_map_wjy.sh --growth
```

Open a specific saved result:

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/<time_dir> \
  --growth
```

Open the latest offline mapping result:

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_offline_mapping \
  --growth
```

For offline results, the viewer can recover the original `stereo_sequence` from `config.yaml` when that path still exists.

This replays the growing map points, a single yellow trajectory line, camera pose, and saved images when `stereo_sequence/` exists. If the result was not recorded with `--record-sequence`, only the preview image can be shown.

For smoother playback on large maps:

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/<time_dir> \
  --growth \
  --every 2 \
  --image-every 2 \
  --max-points 150000
```

For quick capture-only testing:

```bash
./run_decxin3261v_live_wjy.sh \
  --capture-only \
  --max-frames 5 \
  --preview-dir DECXIN-3261V-message/probe/live_split \
  --save-preview-every 1
```

## 7. Export Map

```bash
Scripts/AdHoc/DECXIN3261V/export_map.sh --result <result-folder>
```

The exporter writes `vo_points_*.ply`, `trajectory.ply`, and, when the run used mapping mode, `mapping_points_*.ply`.
