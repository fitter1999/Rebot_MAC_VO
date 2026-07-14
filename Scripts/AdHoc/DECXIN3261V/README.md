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

### Legacy stability comparison

If the current `640x480 -> 640x480` pipeline drifts more than the earlier stable run, first compare against the legacy chain that matched the stable 07_13 quality tests:

```bash
./run_decxin3261v_legacy_quality_wjy.sh --useRR
```

Legacy realtime mapping:

```bash
./run_decxin3261v_legacy_mapping_wjy.sh --useRR
```

The legacy chain uses the old `Calibration/decxin3261v_screen` rectification, `640x400` rectified images, and `480x300` VO input. It is useful for A/B diagnosis, but it is not the default geometry-accurate `640x480` pipeline.

To save frame images for later Rerun replay, record the sequence while mapping:

```bash
./run_decxin3261v_mapping_wjy.sh \
  --useRR \
  --record-sequence
```

For the most stable outdoor workflow, first record rectified stereo images at high frame rate, then reconstruct offline. Capture-only mode does not run MAC-VO, so it is less sensitive to GPU power state and does not show a live trajectory:

```bash
./run_decxin3261v_live_wjy.sh \
  --capture-only \
  --record-sequence \
  --vo-fps 30 \
  --status-every 30
```

Stop with `Ctrl+C`, then check stereo geometry from the saved capture folder:

```bash
./run_macvo_wjy.sh python Scripts/AdHoc/DECXIN3261V/Diagnose_Sequence.py \
  --result Results_decxin3261v_live/<time_dir> \
  --every 30
```

Good captures usually have `dy_mad` close to `0`, `disp_pos_ratio` close to `1`, and `sgbm_valid_ratio` above roughly `0.65`. If `dy_mad` is often above `0.5 px`, fix the camera mount, cable strain, and field of view before reconstructing. Avoid seeing the laptop body, hand, cable, or other objects attached to the camera.

Recommended offline reconstruction uses 3 Hz legacy mapping. This is usually steadier than full 30 Hz replay for long hand-held paths:

```bash
./run_decxin3261v_offline_mapping_wjy.sh \
  --result Results_decxin3261v_live/<time_dir> \
  --target-fps 3 \
  --odom Config/Experiment/MACVO/MACVO_DECXIN3261V_LegacyMapping.yaml \
  --timing
```

For default mapping comparison:

```bash
./run_decxin3261v_offline_mapping_wjy.sh \
  --result Results_decxin3261v_live/<time_dir> \
  --target-fps 3 \
  --timing
```

For full-frame replay, omit `--target-fps`. It feeds every captured frame to MAC-VO and can produce more map points, but may drift more than the 3 Hz version:

```bash
./run_decxin3261v_offline_mapping_wjy.sh \
  --result Results_decxin3261v_live/<time_dir> \
  --timing
```

After reconstruction, use the generated result directory to replay map growth, the yellow trajectory, camera poses, and images in Rerun:

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_offline_mapping/<project_name>/<result_time> \
  --growth
```

If diagnosis or mapping reports `Invalid stereo_sequence`, count both sides:

```bash
find Results_decxin3261v_live/<time_dir>/stereo_sequence/left -maxdepth 1 -name '*.png' | wc -l
find Results_decxin3261v_live/<time_dir>/stereo_sequence/right -maxdepth 1 -name '*.png' | wc -l
```

If only one unpaired final image exists, remove that unpaired image and rerun diagnosis.

Offline mapping defaults to `Config/Experiment/MACVO/MACVO_DECXIN3261V_Mapping.yaml`, not `MACVO_DECXIN3261V_Quality.yaml`. Both use the same `MACVO_FrontendCov.pth` frontend model and similar quality-oriented parameters, but the Mapping config enables `mapping: true` and `mapping_num_point: 2000`; the Quality config has `mapping: false`.

New capture folders include `stereo_sequence/metadata.yaml`. Offline mapping reads this file first, so the replay uses the same rectified camera intrinsics and baseline as the capture. Older capture-only folders do not have this metadata; the offline script prints the fallback camera it uses.

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
