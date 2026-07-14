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

For quick capture-only testing:

```bash
./run_decxin3261v_live_wjy.sh \
  --capture-only \
  --max-frames 5 \
  --preview-dir DECXIN-3261V-message/probe/live_split \
  --save-preview-every 1
```

## 6. Export Map

```bash
Scripts/AdHoc/DECXIN3261V/export_map.sh --result <result-folder>
```

The exporter writes `vo_points_*.ply`, `trajectory.ply`, and, when the run used mapping mode, `mapping_points_*.ply`.
