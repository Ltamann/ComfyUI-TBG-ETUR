import torch


def _drift_log(message):
    try:
        print(f"[TBG Drift Correction] {message}")
    except Exception:
        pass


def sift_tensor_to_uint8_rgb(image):
    if image is None or not torch.is_tensor(image):
        return None
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or int(image.shape[-1]) < 3:
        return None
    image = image.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    image = image[..., :3]
    return (image.numpy() * 255.0 + 0.5).astype("uint8")


def _resize_bhwc_like(reference_bhwc, sampled_bhwc):
    if not torch.is_tensor(reference_bhwc) or not torch.is_tensor(sampled_bhwc):
        return reference_bhwc
    if reference_bhwc.ndim != 4 or sampled_bhwc.ndim != 4:
        return reference_bhwc
    ref_h = int(reference_bhwc.shape[1])
    ref_w = int(reference_bhwc.shape[2])
    sample_h = int(sampled_bhwc.shape[1])
    sample_w = int(sampled_bhwc.shape[2])
    if ref_h == sample_h and ref_w == sample_w:
        return reference_bhwc
    return torch.nn.functional.interpolate(
        reference_bhwc.permute(0, 3, 1, 2).to(dtype=torch.float32),
        size=(sample_h, sample_w),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous().to(device=reference_bhwc.device, dtype=reference_bhwc.dtype)


def _resize_rgb_like(reference_rgb, sampled_rgb):
    import cv2

    if reference_rgb is None or sampled_rgb is None:
        return reference_rgb
    if not hasattr(reference_rgb, "shape") or not hasattr(sampled_rgb, "shape"):
        return reference_rgb
    if len(reference_rgb.shape) < 2 or len(sampled_rgb.shape) < 2:
        return reference_rgb
    ref_h = int(reference_rgb.shape[0])
    ref_w = int(reference_rgb.shape[1])
    sample_h = int(sampled_rgb.shape[0])
    sample_w = int(sampled_rgb.shape[1])
    if ref_h == sample_h and ref_w == sample_w:
        return reference_rgb
    interpolation = cv2.INTER_AREA if sample_h < ref_h or sample_w < ref_w else cv2.INTER_LINEAR
    return cv2.resize(reference_rgb, (sample_w, sample_h), interpolation=interpolation)


def _apply_sift_affine_correction(reference_tile, sampled_tile, index=None, enhanced=False, level=1):
    info = {
        "changed": False,
        "reason": "disabled_or_unavailable",
        "matches": 0,
        "inliers": 0,
    }
    if reference_tile is None or sampled_tile is None:
        info["reason"] = "missing_input"
        return sampled_tile, info
    if not torch.is_tensor(reference_tile) or not torch.is_tensor(sampled_tile):
        info["reason"] = "non_tensor_input"
        return sampled_tile, info

    sampled_original_ndim = sampled_tile.ndim
    sampled_bhwc = sampled_tile.unsqueeze(0) if sampled_tile.ndim == 3 else sampled_tile
    reference_bhwc = reference_tile.unsqueeze(0) if reference_tile.ndim == 3 else reference_tile
    if sampled_bhwc.ndim != 4 or reference_bhwc.ndim != 4:
        info["reason"] = "invalid_tensor_shape"
        return sampled_tile, info
    if tuple(sampled_bhwc.shape[:2]) != tuple(reference_bhwc.shape[:2]) or tuple(sampled_bhwc.shape[3:]) != tuple(reference_bhwc.shape[3:]):
        reference_bhwc = _resize_bhwc_like(reference_bhwc, sampled_bhwc)
        info["reference_resized"] = True
        _drift_log(
            f"{'tile ' + str(index + 1) if isinstance(index, int) else 'standalone'}: "
            f"resized reference from {tuple(reference_tile.shape)} to {tuple(reference_bhwc.shape)}"
        )
    if tuple(sampled_bhwc.shape) != tuple(reference_bhwc.shape):
        info["reason"] = "shape_mismatch"
        return sampled_tile, info
    if int(sampled_bhwc.shape[-1]) < 3 or int(reference_bhwc.shape[-1]) < 3:
        info["reason"] = "not_rgb"
        return sampled_tile, info

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        info["reason"] = f"opencv_numpy_unavailable:{type(exc).__name__}"
        return sampled_tile, info
    if not hasattr(cv2, "SIFT_create"):
        info["reason"] = "sift_unavailable"
        return sampled_tile, info

    reference_np = sift_tensor_to_uint8_rgb(reference_bhwc)
    sampled_np = sift_tensor_to_uint8_rgb(sampled_bhwc)
    if reference_np is None or sampled_np is None:
        info["reason"] = "conversion_failed"
        return sampled_tile, info

    corrected_images = []
    any_changed = False
    total_matches = 0
    total_inliers = 0
    last_reason = "no_batches"
    def make_attempts():
        attempts = [
            {
                "name": "sift_rgb_luma",
                "feature": lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                "sift_kwargs": {
                    "nfeatures": 0,
                    "nOctaveLayers": 3,
                    "contrastThreshold": 0.01,
                    "edgeThreshold": 20,
                    "sigma": 1.6,
                },
                "ratio": 0.80,
                "ransac": 4.0,
                "estimator": "partial",
                "min_inlier_scale": 0.012,
                "min_inliers_floor": 24,
                "min_inlier_ratio": 0.06,
            },
        ]
        if enhanced:
            def highpass_feature(rgb):
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
                highpass = cv2.addWeighted(gray, 2.0, blur, -1.0, 0)
                return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(highpass)

            def highpass_wide_feature(rgb):
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (0, 0), 7.0)
                highpass = cv2.addWeighted(gray, 2.0, blur, -1.0, 0)
                return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(highpass)

            def edge_feature(rgb):
                highpass = highpass_feature(rgb)
                median = float(np.median(highpass))
                low = int(max(0, 0.50 * median))
                high = int(min(255, 1.60 * median))
                edges = cv2.Canny(highpass, low, high)
                return cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

            attempts.extend([
                {
                    "name": "sift_highpass",
                    "feature": highpass_feature,
                    "sift_kwargs": {
                        "nfeatures": 12000,
                        "nOctaveLayers": 5,
                        "contrastThreshold": 0.004,
                        "edgeThreshold": 35,
                        "sigma": 1.2,
                    },
                    "ratio": 0.86,
                    "ransac": 5.0,
                    "estimator": "partial",
                    "min_inlier_scale": 0.008,
                    "min_inliers_floor": 14,
                    "min_inlier_ratio": 0.02,
                },
                {
                    "name": "sift_highpass_wide",
                    "feature": highpass_wide_feature,
                    "sift_kwargs": {
                        "nfeatures": 20000,
                        "nOctaveLayers": 5,
                        "contrastThreshold": 0.002,
                        "edgeThreshold": 40,
                        "sigma": 1.2,
                    },
                    "ratio": 0.90,
                    "ransac": 5.0,
                    "estimator": "partial",
                    "min_inlier_scale": 0.007,
                    "min_inliers_floor": 12,
                    "min_inlier_ratio": 0.015,
                },
                {
                    "name": "akaze_highpass_mldb",
                    "detector": "akaze_mldb",
                    "feature": highpass_wide_feature,
                    "ratio": 0.90,
                    "ransac": 5.0,
                    "estimator": "partial",
                    "min_inlier_scale": 0.005,
                    "min_inliers_floor": 8,
                    "min_inlier_ratio": 0.040,
                },
                {
                    "name": "akaze_highpass_mldb_full_affine",
                    "detector": "akaze_mldb",
                    "feature": highpass_wide_feature,
                    "ratio": 0.90,
                    "ransac": 5.0,
                    "estimator": "full_affine",
                    "min_level": 3,
                    "min_inlier_scale": 0.005,
                    "min_inliers_floor": 8,
                    "min_inlier_ratio": 0.040,
                    "max_shear": 0.18,
                },
                {
                    "name": "sift_highpass_wide_rootsift",
                    "feature": highpass_wide_feature,
                    "root_sift": True,
                    "sift_kwargs": {
                        "nfeatures": 20000,
                        "nOctaveLayers": 5,
                        "contrastThreshold": 0.004,
                        "edgeThreshold": 40,
                        "sigma": 1.2,
                    },
                    "ratio": 0.85,
                    "ransac": 5.0,
                    "estimator": "partial",
                    "min_inlier_scale": 0.007,
                    "min_inliers_floor": 12,
                    "min_inlier_ratio": 0.050,
                },
                {
                    "name": "sift_highpass_wide_rootsift_full_affine",
                    "feature": highpass_wide_feature,
                    "root_sift": True,
                    "sift_kwargs": {
                        "nfeatures": 20000,
                        "nOctaveLayers": 5,
                        "contrastThreshold": 0.004,
                        "edgeThreshold": 40,
                        "sigma": 1.2,
                    },
                    "ratio": 0.85,
                    "ransac": 5.0,
                    "estimator": "full_affine",
                    "min_level": 3,
                    "min_inlier_scale": 0.007,
                    "min_inliers_floor": 12,
                    "min_inlier_ratio": 0.050,
                    "max_shear": 0.18,
                },
                {
                    "name": "sift_border_highpass",
                    "feature": highpass_wide_feature,
                    "mask": "border",
                    "sift_kwargs": {
                        "nfeatures": 16000,
                        "nOctaveLayers": 5,
                        "contrastThreshold": 0.002,
                        "edgeThreshold": 45,
                        "sigma": 1.2,
                    },
                    "ratio": 0.92,
                    "ransac": 6.0,
                    "estimator": "partial",
                    "min_inlier_scale": 0.005,
                    "min_inliers_floor": 8,
                    "min_inlier_ratio": 0.010,
                },
                {
                    "name": "sift_border_roi_rootsift",
                    "feature": highpass_wide_feature,
                    "root_sift": True,
                    "roi": "border",
                    "sift_kwargs": {
                        "nfeatures": 8000,
                        "nOctaveLayers": 5,
                        "contrastThreshold": 0.003,
                        "edgeThreshold": 45,
                        "sigma": 1.2,
                    },
                    "ratio": 0.85,
                    "ransac": 5.0,
                    "estimator": "partial",
                    "min_inlier_scale": 0.005,
                    "min_inliers_floor": 10,
                    "min_inlier_ratio": 0.040,
                },
                {
                    "name": "akaze_border_roi_mldb",
                    "detector": "akaze_mldb",
                    "feature": highpass_wide_feature,
                    "roi": "border",
                    "ratio": 0.92,
                    "ransac": 6.0,
                    "estimator": "partial",
                    "min_inlier_scale": 0.004,
                    "min_inliers_floor": 6,
                    "min_inlier_ratio": 0.020,
                },
                {
                    "name": "sift_contours",
                    "feature": edge_feature,
                    "sift_kwargs": {
                        "nfeatures": 16000,
                        "nOctaveLayers": 5,
                        "contrastThreshold": 0.003,
                        "edgeThreshold": 45,
                        "sigma": 1.1,
                    },
                    "ratio": 0.88,
                    "ransac": 5.5,
                    "estimator": "partial",
                    "min_inlier_scale": 0.007,
                    "min_inliers_floor": 12,
                    "min_inlier_ratio": 0.02,
                },
            ])
        return attempts

    attempts = make_attempts()

    for batch_index in range(int(sampled_np.shape[0])):
        ref_rgb = reference_np[batch_index]
        sample_rgb = sampled_np[batch_index]
        height, width = ref_rgb.shape[:2]
        if height < 16 or width < 16:
            corrected_images.append(sample_rgb)
            last_reason = "image_too_small"
            continue
        ref_luma = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
        sample_luma = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)
        accepted = None
        batch_last_reason = "no_sift_attempts"
        batch_matches = 0
        batch_inliers = 0
        for attempt in attempts:
            if int(level) < int(attempt.get("min_level", 1)):
                continue
            attempt_name = attempt["name"]
            try:
                if attempt.get("detector") == "akaze_mldb":
                    detector = cv2.AKAZE_create(
                        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                        threshold=0.0005,
                        nOctaves=4,
                        nOctaveLayers=4,
                    )
                    matcher_norm = cv2.NORM_HAMMING
                else:
                    detector = cv2.SIFT_create(**attempt["sift_kwargs"])
                    matcher_norm = cv2.NORM_L2
                if attempt.get("roi") == "border":
                    roi_band = int(max(64, min(height // 2, width // 2, round(float(min(height, width)) * 0.18))))
                    rois = [
                        (0, 0, width, roi_band),
                        (0, height - roi_band, width, roi_band),
                        (0, 0, roi_band, height),
                        (width - roi_band, 0, roi_band, height),
                    ]
                    keypoints_ref = []
                    keypoints_sample = []
                    descriptors_ref_parts = []
                    descriptors_sample_parts = []
                    for roi_x, roi_y, roi_w, roi_h in rois:
                        ref_gray = attempt["feature"](ref_rgb[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])
                        sample_gray = attempt["feature"](sample_rgb[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])
                        roi_keypoints_ref, roi_descriptors_ref = detector.detectAndCompute(ref_gray, None)
                        roi_keypoints_sample, roi_descriptors_sample = detector.detectAndCompute(sample_gray, None)
                        if roi_descriptors_ref is not None and roi_keypoints_ref:
                            for keypoint in roi_keypoints_ref:
                                keypoint.pt = (keypoint.pt[0] + float(roi_x), keypoint.pt[1] + float(roi_y))
                            keypoints_ref.extend(roi_keypoints_ref)
                            descriptors_ref_parts.append(roi_descriptors_ref)
                        if roi_descriptors_sample is not None and roi_keypoints_sample:
                            for keypoint in roi_keypoints_sample:
                                keypoint.pt = (keypoint.pt[0] + float(roi_x), keypoint.pt[1] + float(roi_y))
                            keypoints_sample.extend(roi_keypoints_sample)
                            descriptors_sample_parts.append(roi_descriptors_sample)
                    descriptors_ref = np.concatenate(descriptors_ref_parts, axis=0) if descriptors_ref_parts else None
                    descriptors_sample = np.concatenate(descriptors_sample_parts, axis=0) if descriptors_sample_parts else None
                else:
                    ref_gray = attempt["feature"](ref_rgb)
                    sample_gray = attempt["feature"](sample_rgb)
                    feature_mask = None
                    if attempt.get("mask") == "border":
                        feature_mask = _sift_border_mask(height, width, scale=0.22)
                    keypoints_ref, descriptors_ref = detector.detectAndCompute(ref_gray, feature_mask)
                    keypoints_sample, descriptors_sample = detector.detectAndCompute(sample_gray, feature_mask)
            except Exception as exc:
                batch_last_reason = f"{attempt_name}:feature_detection_failed:{type(exc).__name__}"
                continue

            if (
                    descriptors_ref is None
                    or descriptors_sample is None
                    or len(keypoints_ref) < 10
                    or len(keypoints_sample) < 10
            ):
                batch_last_reason = f"{attempt_name}:too_few_features"
                continue
            if attempt.get("root_sift"):
                descriptors_ref = descriptors_ref.astype(np.float32)
                descriptors_sample = descriptors_sample.astype(np.float32)
                descriptors_ref /= descriptors_ref.sum(axis=1, keepdims=True) + 1.0e-7
                descriptors_sample /= descriptors_sample.sum(axis=1, keepdims=True) + 1.0e-7
                descriptors_ref = np.sqrt(descriptors_ref)
                descriptors_sample = np.sqrt(descriptors_sample)

            try:
                matcher = cv2.BFMatcher(matcher_norm)
                raw_matches = matcher.knnMatch(descriptors_sample, descriptors_ref, k=2)
            except Exception as exc:
                batch_last_reason = f"{attempt_name}:matching_failed:{type(exc).__name__}"
                continue

            good_matches = []
            for pair in raw_matches:
                if len(pair) != 2:
                    continue
                first, second = pair
                if first.distance < float(attempt["ratio"]) * second.distance:
                    good_matches.append(first)
            batch_matches = max(batch_matches, len(good_matches))
            if len(good_matches) < 10:
                batch_last_reason = f"{attempt_name}:too_few_matches"
                continue

            sample_points = np.float32([keypoints_sample[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            ref_points = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            try:
                if attempt.get("estimator") == "full_affine":
                    affine, inlier_mask = cv2.estimateAffine2D(
                        sample_points,
                        ref_points,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=float(attempt["ransac"]),
                        maxIters=8000,
                        confidence=0.997,
                        refineIters=15,
                    )
                    estimator_name = "full_affine"
                else:
                    affine, inlier_mask = cv2.estimateAffinePartial2D(
                        sample_points,
                        ref_points,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=float(attempt["ransac"]),
                        maxIters=6000 if enhanced else 3000,
                        confidence=0.997 if enhanced else 0.995,
                        refineIters=15 if enhanced else 10,
                    )
                    estimator_name = "partial_affine"
            except Exception as exc:
                batch_last_reason = f"{attempt_name}:affine_failed:{type(exc).__name__}"
                continue
            if affine is None or inlier_mask is None:
                batch_last_reason = f"{attempt_name}:affine_none"
                continue

            inliers = int(inlier_mask.ravel().sum())
            batch_inliers = max(batch_inliers, inliers)
            min_side = max(1, min(int(width), int(height)))
            min_inliers = max(
                int(attempt["min_inliers_floor"]),
                min(80, int(round(float(min_side) * float(attempt["min_inlier_scale"])))),
            )
            inlier_ratio = float(inliers) / max(1.0, float(len(good_matches)))
            if inliers < min_inliers:
                batch_last_reason = f"{attempt_name}:too_few_inliers:{inliers}<{min_inliers}"
                continue
            if inlier_ratio < float(attempt["min_inlier_ratio"]):
                batch_last_reason = f"{attempt_name}:low_inlier_ratio:{inlier_ratio:.4f}"
                continue

            inlier_points = sample_points[inlier_mask.ravel().astype(bool)]
            if inlier_points.shape[0] < 3:
                batch_last_reason = f"{attempt_name}:too_few_inlier_points"
                continue
            spread_x = float(inlier_points[:, 0].max() - inlier_points[:, 0].min())
            spread_y = float(inlier_points[:, 1].max() - inlier_points[:, 1].min())
            min_spread = 0.10 if enhanced else 0.12
            if spread_x < float(width) * min_spread or spread_y < float(height) * min_spread:
                batch_last_reason = f"{attempt_name}:poor_inlier_spread:{spread_x:.1f}x{spread_y:.1f}"
                continue

            affine = affine.astype(np.float32)
            a, b, tx = [float(v) for v in affine[0]]
            c, d, ty = [float(v) for v in affine[1]]
            scale_x = float((a * a + c * c) ** 0.5)
            scale_y = float((b * b + d * d) ** 0.5)
            shear = abs(float((a * b + c * d) / max(1.0e-6, scale_x * scale_y)))
            rotation_deg = abs(float(np.degrees(np.arctan2(c, a))))
            translation = float((tx * tx + ty * ty) ** 0.5)
            if scale_x < 0.94 or scale_x > 1.06 or scale_y < 0.94 or scale_y > 1.06:
                batch_last_reason = f"{attempt_name}:unsafe_scale:{scale_x:.4f},{scale_y:.4f}"
                continue
            if rotation_deg > 5.0:
                batch_last_reason = f"{attempt_name}:unsafe_rotation:{rotation_deg:.3f}"
                continue
            max_shear = float(attempt.get("max_shear", 0.08))
            if shear > max_shear:
                batch_last_reason = f"{attempt_name}:unsafe_shear:{shear:.4f}"
                continue
            if translation > float(min_side) * 0.08:
                batch_last_reason = f"{attempt_name}:unsafe_translation:{translation:.2f}"
                continue

            corners = np.float32(
                [
                    [0.0, 0.0],
                    [float(width - 1), 0.0],
                    [float(width - 1), float(height - 1)],
                    [0.0, float(height - 1)],
                ]
            ).reshape(-1, 1, 2)
            warped_corners = cv2.transform(corners, affine).reshape(-1, 2)
            corner_shift = float(np.linalg.norm(warped_corners - corners.reshape(-1, 2), axis=1).max())
            if corner_shift > float(min_side) * 0.08:
                batch_last_reason = f"{attempt_name}:unsafe_corner_shift:{corner_shift:.2f}"
                continue

            try:
                corrected = cv2.warpAffine(
                    sample_rgb,
                    affine,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            except Exception as exc:
                batch_last_reason = f"{attempt_name}:warp_failed:{type(exc).__name__}"
                continue

            try:
                ref_luma_f = ref_luma.astype(np.float32)
                before_luma = sample_luma.astype(np.float32)
                after_luma = cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY).astype(np.float32)
                before_mad = float(np.mean(np.abs(before_luma - ref_luma_f)))
                after_mad = float(np.mean(np.abs(after_luma - ref_luma_f)))
                if after_mad > before_mad * 1.08:
                    batch_last_reason = f"{attempt_name}:mad_regression:{after_mad:.4f}>{before_mad:.4f}"
                    continue
            except Exception:
                pass

            accepted = (
                corrected,
                (
                    f"corrected_affine estimator={estimator_name} attempt={attempt_name} inlier_ratio={inlier_ratio:.4f} "
                    f"spread={spread_x:.1f}x{spread_y:.1f} "
                    f"scale={scale_x:.4f},{scale_y:.4f} "
                    f"rot={rotation_deg:.3f} shear={shear:.4f} shift={translation:.2f} corner={corner_shift:.2f}"
                ),
                len(good_matches),
                inliers,
            )
            break

        total_matches += batch_matches
        total_inliers += batch_inliers
        if accepted is None:
            if enhanced:
                corrected, phase_info = _repeat_translation_alignment(ref_rgb, sample_rgb)
                corrected_images.append(corrected)
                any_changed = any_changed or bool(phase_info.get("changed", False))
                last_reason = f"{phase_info.get('reason', 'phase_translation')} after sift:{batch_last_reason}"
            else:
                corrected_images.append(sample_rgb)
                last_reason = batch_last_reason
            continue

        corrected_images.append(accepted[0])
        any_changed = True
        last_reason = accepted[1]

    if not any_changed:
        info.update({
            "reason": last_reason,
            "matches": total_matches,
            "inliers": total_inliers,
        })
        return sampled_tile, info

    corrected_np = np.stack(corrected_images, axis=0)
    corrected_tensor = torch.from_numpy(corrected_np).to(
        device=sampled_bhwc.device,
        dtype=torch.float32,
    ) / 255.0
    corrected_tensor = corrected_tensor.to(dtype=sampled_bhwc.dtype)
    if int(sampled_bhwc.shape[-1]) > 3:
        corrected_tensor = torch.cat([corrected_tensor, sampled_bhwc[..., 3:]], dim=-1)
    if sampled_original_ndim == 3:
        corrected_tensor = corrected_tensor[0]
    info.update({
        "changed": True,
        "reason": last_reason,
        "matches": total_matches,
        "inliers": total_inliers,
    })
    return corrected_tensor.clamp(0.0, 1.0), info


DRIFT_CORRECTION_MODES = [
    "Off",
    "On",
]


def normalize_drift_correction_mode(value):
    if isinstance(value, bool):
        return "x4" if value else "Off"
    key = str(value or "").strip().lower()
    if key in ("", "none", "off", "false", "0", "disabled"):
        return "Off"
    if key in ("true", "1", "on", "enabled", "sift", "scale-invariant feature transform"):
        return "x4"
    if key in ("x1",):
        return "x1"
    if key in ("3", "x3"):
        return "x3"
    if key in ("4", "x4", "max", "maximum"):
        return "x4"
    if key in ("2", "x2") or "flow" in key or "fallback" in key:
        return "x2"
    return "Off"


def _drift_mode_level(mode):
    mode = normalize_drift_correction_mode(mode)
    if mode == "Off":
        return 0
    try:
        return int(mode[1:])
    except Exception:
        return 1


def _sift_border_mask(height, width, scale=0.16):
    import cv2
    import numpy as np

    band = int(round(float(min(height, width)) * float(scale)))
    band = max(16, min(band, height // 2, width // 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:band, :] = 255
    mask[-band:, :] = 255
    mask[:, :band] = 255
    mask[:, -band:] = 255
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, float(band) * 0.10))


def _border_weighted_phase_translation(reference_rgb, sample_rgb):
    import cv2
    import numpy as np

    height, width = reference_rgb.shape[:2]
    if height < 16 or width < 16:
        return sample_rgb, {
            "changed": False,
            "reason": "phase_translation_image_too_small",
        }

    def feature(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(gray, (0, 0), 5.0)
        highpass = gray * 2.0 - blur
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)
        edge = edge / (float(edge.max()) + 1.0e-6)
        out = np.clip(highpass * 0.65 + edge * 0.35, 0.0, 1.0)
        border = _sift_border_mask(height, width, scale=0.20).astype(np.float32) / 255.0
        return (out * (0.35 + 0.65 * border)).astype(np.float32)

    ref_feature = feature(reference_rgb)
    sample_feature = feature(sample_rgb)
    try:
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(ref_feature, sample_feature, window)
    except Exception as exc:
        return sample_rgb, {
            "changed": False,
            "reason": f"phase_translation_failed:{type(exc).__name__}",
        }

    min_side = max(1, min(int(height), int(width)))
    cap = float(min_side) * 0.08
    shift_mag = float((dx * dx + dy * dy) ** 0.5)
    capped = False
    if shift_mag > cap:
        scale = cap / max(shift_mag, 1.0e-6)
        dx *= scale
        dy *= scale
        capped = True

    affine = np.float32([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]])
    corrected = cv2.warpAffine(
        sample_rgb,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return corrected, {
        "changed": True,
        "reason": (
            f"phase_border_translation dx={-float(dx):.2f} dy={-float(dy):.2f} "
            f"response={float(response):.4f} capped={capped}"
        ),
    }


def _weighted_translation_search(reference_rgb, sample_rgb, border_priority=True):
    import cv2
    import numpy as np

    height, width = reference_rgb.shape[:2]
    if height < 16 or width < 16:
        return sample_rgb, {
            "changed": False,
            "reason": "translation_search_image_too_small",
        }

    def features(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(gray, (0, 0), 5.0)
        highpass = np.clip(gray * 2.0 - blur, 0.0, 1.0)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)
        edge = edge / (float(edge.max()) + 1.0e-6)
        return highpass, edge, gray

    ref_high, ref_edge, ref_gray = features(reference_rgb)
    sample_high, sample_edge, sample_gray = features(sample_rgb)
    if border_priority:
        border = _sift_border_mask(height, width, scale=0.20).astype(np.float32) / 255.0
        weight = 0.20 + 0.80 * border
        label = "border_translation_search"
    else:
        weight = np.ones((height, width), dtype=np.float32)
        label = "global_translation_search"
    max_shift = int(max(8, min(48, round(float(min(height, width)) * 0.047))))
    best = None
    for step in (4, 2, 1):
        if best is None:
            x_values = range(-max_shift, max_shift + 1, step)
            y_values = range(-max_shift, max_shift + 1, step)
        else:
            _, best_dx, best_dy = best
            x_values = range(best_dx - 4 * step, best_dx + 4 * step + 1, step)
            y_values = range(best_dy - 4 * step, best_dy + 4 * step + 1, step)
        for dx in x_values:
            for dy in y_values:
                if abs(dx) > max_shift or abs(dy) > max_shift:
                    continue
                affine = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
                warped_high = cv2.warpAffine(sample_high, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                warped_edge = cv2.warpAffine(sample_edge, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                warped_gray = cv2.warpAffine(sample_gray, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                score = float(((
                    np.abs(warped_high - ref_high) * 0.45
                    + np.abs(warped_edge - ref_edge) * 0.45
                    + np.abs(warped_gray - ref_gray) * 0.10
                ) * weight).mean())
                score += float((dx * dx + dy * dy) ** 0.5) * 0.0008
                if best is None or score < best[0]:
                    best = (score, int(dx), int(dy))

    _, dx, dy = best or (0.0, 0, 0)
    affine = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    corrected = cv2.warpAffine(
        sample_rgb,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return corrected, {
        "changed": True,
        "reason": f"{label} dx={dx:.2f} dy={dy:.2f} max={max_shift}",
    }


def _repeat_translation_alignment(reference_rgb, sample_rgb):
    import numpy as np

    best = None
    for border_priority in (True, False):
        corrected, info = _weighted_translation_search(reference_rgb, sample_rgb, border_priority=border_priority)
        try:
            masks = _border_masks(reference_rgb.shape[0], reference_rgb.shape[1])
            metrics = _alignment_metrics(corrected, reference_rgb, masks)
            score = metrics["luma_border"] * 2.0 + metrics["edge_border"] + metrics["luma_all"] * 0.5
        except Exception:
            score = float(np.mean(np.abs(corrected.astype(np.float32) - reference_rgb.astype(np.float32))))
        if best is None or score < best[0]:
            best = (score, corrected, info)
    return best[1], best[2]


def _luma_mad(a, b, mask=None):
    import cv2
    import numpy as np

    if hasattr(a, "shape") and hasattr(b, "shape") and tuple(a.shape[:2]) != tuple(b.shape[:2]):
        b = _resize_rgb_like(b, a)
    a_luma = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b_luma = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    delta = np.abs(a_luma - b_luma)
    if mask is not None and np.any(mask > 0):
        delta = delta[mask > 0]
    return float(delta.mean()) if delta.size else float("inf")


def _edge_mad(a, b, mask=None):
    import cv2
    import numpy as np

    if hasattr(a, "shape") and hasattr(b, "shape") and tuple(a.shape[:2]) != tuple(b.shape[:2]):
        b = _resize_rgb_like(b, a)

    def edge_mag(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    delta = np.abs(edge_mag(a) - edge_mag(b))
    if mask is not None and np.any(mask > 0):
        delta = delta[mask > 0]
    return float(delta.mean()) if delta.size else float("inf")


def _border_masks(height, width, band=None):
    import cv2
    import numpy as np

    band = int(band or max(24, min(96, round(min(height, width) * 0.094))))
    band = max(1, min(band, height // 2, width // 2))
    masks = {}
    for name in ("top", "bottom", "left", "right", "border"):
        masks[name] = np.zeros((height, width), dtype=np.uint8)
    masks["top"][:band, :] = 255
    masks["bottom"][-band:, :] = 255
    masks["left"][:, :band] = 255
    masks["right"][:, -band:] = 255
    masks["border"] = cv2.bitwise_or(
        cv2.bitwise_or(masks["top"], masks["bottom"]),
        cv2.bitwise_or(masks["left"], masks["right"]),
    )
    return masks


def _alignment_metrics(sample_rgb, ref_rgb, masks):
    if hasattr(sample_rgb, "shape") and hasattr(ref_rgb, "shape"):
        if tuple(sample_rgb.shape[:2]) != tuple(ref_rgb.shape[:2]):
            ref_rgb = _resize_rgb_like(ref_rgb, sample_rgb)
    metrics = {
        "luma_all": _luma_mad(sample_rgb, ref_rgb),
        "edge_all": _edge_mad(sample_rgb, ref_rgb),
        "luma_border": _luma_mad(sample_rgb, ref_rgb, masks["border"]),
        "edge_border": _edge_mad(sample_rgb, ref_rgb, masks["border"]),
    }
    for side in ("top", "bottom", "left", "right"):
        metrics[f"luma_{side}"] = _luma_mad(sample_rgb, ref_rgb, masks[side])
        metrics[f"edge_{side}"] = _edge_mad(sample_rgb, ref_rgb, masks[side])
    return metrics


def _strict_border_accept(candidate, baseline):
    if candidate["luma_all"] >= baseline["luma_all"]:
        return False
    for side in ("top", "bottom", "left", "right"):
        if candidate[f"luma_{side}"] > baseline[f"luma_{side}"] * 1.02:
            return False
        if candidate[f"edge_{side}"] > baseline[f"edge_{side}"] * 1.08:
            return False
    return True


def _candidate_score(candidate, baseline):
    return (
        2.0 * candidate["luma_border"] / max(1.0e-6, baseline["luma_border"])
        + 2.0 * candidate["edge_border"] / max(1.0e-6, baseline["edge_border"])
        + candidate["luma_all"] / max(1.0e-6, baseline["luma_all"])
        + candidate["edge_all"] / max(1.0e-6, baseline["edge_all"])
    )


def _gray_feature(rgb, mode):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if mode == "gray":
        return gray
    if mode == "highpass":
        blur = cv2.GaussianBlur(gray, (0, 0), 5.0)
        highpass = cv2.addWeighted(gray, 2.0, blur, -1.0, 0)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(highpass)
    if mode == "edge":
        highpass = _gray_feature(rgb, "highpass")
        median = float(np.median(highpass))
        low = int(max(0, 0.55 * median))
        high = int(min(255, 1.45 * median))
        return cv2.dilate(cv2.Canny(highpass, low, high), np.ones((3, 3), dtype=np.uint8), iterations=1)
    return gray


def _remap_reverse_flow(sample_rgb, flow, sigma=0, cap=32):
    import cv2
    import numpy as np

    height, width = sample_rgb.shape[:2]
    work = flow.astype(np.float32, copy=True)
    if sigma and sigma > 0:
        work[..., 0] = cv2.GaussianBlur(work[..., 0], (0, 0), float(sigma))
        work[..., 1] = cv2.GaussianBlur(work[..., 1], (0, 0), float(sigma))
    if cap and cap > 0:
        magnitude = np.sqrt(work[..., 0] * work[..., 0] + work[..., 1] * work[..., 1])
        work *= np.minimum(1.0, float(cap) / (magnitude + 1.0e-6))[..., None]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    corrected = cv2.remap(
        sample_rgb,
        xx + work[..., 0],
        yy + work[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    magnitude = np.sqrt(work[..., 0] * work[..., 0] + work[..., 1] * work[..., 1])
    grad_y0, grad_x0 = np.gradient(work[..., 0])
    grad_y1, grad_x1 = np.gradient(work[..., 1])
    deformation = np.sqrt(
        grad_x0 * grad_x0
        + grad_y0 * grad_y0
        + grad_x1 * grad_x1
        + grad_y1 * grad_y1
    )
    flow_info = {
        "flow_mean": float(magnitude.mean()),
        "flow_p95": float(np.percentile(magnitude, 95)),
        "flow_max": float(magnitude.max()),
        "flow_deform_p95": float(np.percentile(deformation, 95)),
        "flow_deform_max": float(deformation.max()),
    }
    return corrected, flow_info


def _apply_reverse_flow_fallback(reference_tile, sampled_tile, level=2):
    info = {
        "changed": False,
        "reason": "flow_unavailable",
        "matches": 0,
        "inliers": 0,
    }
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        info["reason"] = f"opencv_numpy_unavailable:{type(exc).__name__}"
        return sampled_tile, info

    sampled_original_ndim = sampled_tile.ndim
    sampled_bhwc = sampled_tile.unsqueeze(0) if sampled_tile.ndim == 3 else sampled_tile
    reference_bhwc = reference_tile.unsqueeze(0) if reference_tile.ndim == 3 else reference_tile
    reference_np = sift_tensor_to_uint8_rgb(reference_bhwc)
    sampled_np = sift_tensor_to_uint8_rgb(sampled_bhwc)
    if reference_np is None or sampled_np is None:
        info["reason"] = "conversion_failed"
        return sampled_tile, info

    corrected_images = []
    any_changed = False
    last_reason = "no_accepted_flow"
    accepted_count = 0
    for batch_index in range(int(sampled_np.shape[0])):
        ref_rgb = reference_np[batch_index]
        sample_rgb = sampled_np[batch_index]
        ref_rgb = _resize_rgb_like(ref_rgb, sample_rgb)
        height, width = ref_rgb.shape[:2]
        if height < 32 or width < 32:
            corrected_images.append(sample_rgb)
            last_reason = "image_too_small"
            continue
        masks = _border_masks(height, width)
        baseline = _alignment_metrics(sample_rgb, ref_rgb, masks)
        include_deepflow = int(level) >= 4
        candidates = []
        rgb_features = [
            ("gray", _gray_feature(ref_rgb, "gray"), _gray_feature(sample_rgb, "gray")),
            ("highpass", _gray_feature(ref_rgb, "highpass"), _gray_feature(sample_rgb, "highpass")),
        ]
        feature_stages = [(rgb_features, "dis")]
        if include_deepflow and hasattr(cv2, "optflow"):
            feature_stages.append((rgb_features, "deepflow"))

        for feature_pairs, engine_stage in feature_stages:
            for feature_name, ref_feature, sample_feature in feature_pairs:
                flow_variants = []
                if engine_stage == "dis":
                    try:
                        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                        flow_variants.append(("dis_medium", dis.calc(ref_feature, sample_feature, None)))
                    except Exception:
                        pass
                elif engine_stage == "deepflow":
                    try:
                        flow_variants.append(("deepflow", cv2.optflow.createOptFlow_DeepFlow().calc(ref_feature, sample_feature, None)))
                    except Exception:
                        pass

                for engine_name, flow in flow_variants:
                    if flow is None or tuple(flow.shape[:2]) != (height, width):
                        continue
                    for sigma in (0, 9):
                        for cap in (16, 32):
                            corrected, flow_info = _remap_reverse_flow(sample_rgb, flow, sigma=sigma, cap=cap)
                            metrics = _alignment_metrics(corrected, ref_rgb, masks)
                            strict = _strict_border_accept(metrics, baseline)
                            aggregate = (
                                metrics["luma_all"] < baseline["luma_all"]
                                and metrics["luma_border"] <= baseline["luma_border"] * 1.01
                                and metrics["edge_border"] <= baseline["edge_border"] * 1.08
                            )
                            metrics.update(flow_info)
                            if metrics.get("flow_deform_p95", 999.0) > 0.56:
                                continue
                            metrics["strict"] = strict
                            metrics["aggregate"] = aggregate
                            metrics["feature"] = feature_name
                            metrics["engine"] = engine_name
                            metrics["sigma"] = sigma
                            metrics["cap"] = cap
                            score = _candidate_score(metrics, baseline) + metrics.get("flow_deform_p95", 0.0) * 0.35
                            improves = score < 6.0
                            if strict or aggregate or improves:
                                candidates.append((strict, aggregate, score, corrected, metrics))
            if any(candidate[0] for candidate in candidates):
                break

        if not candidates:
            corrected_images.append(sample_rgb)
            last_reason = "no_border_safe_flow_candidate"
            continue

        candidates.sort(key=lambda item: (not item[0], not item[1], item[2]))
        _, _, _, best_image, best_metrics = candidates[0]
        corrected_images.append(best_image)
        any_changed = True
        accepted_count += 1
        last_reason = (
            f"corrected_reverse_flow {best_metrics['engine']}:{best_metrics['feature']} "
            f"sigma={best_metrics['sigma']} cap={best_metrics['cap']} "
            f"strict={best_metrics['strict']} aggregate={best_metrics['aggregate']} "
            f"luma_all={baseline['luma_all']:.3f}->{best_metrics['luma_all']:.3f} "
            f"luma_border={baseline['luma_border']:.3f}->{best_metrics['luma_border']:.3f} "
            f"edge_border={baseline['edge_border']:.3f}->{best_metrics['edge_border']:.3f} "
            f"flow_p95={best_metrics.get('flow_p95', 0.0):.2f}"
        )

    if not any_changed:
        info["reason"] = last_reason
        return sampled_tile, info

    corrected_np = np.stack(corrected_images, axis=0)
    corrected_tensor = torch.from_numpy(corrected_np).to(
        device=sampled_bhwc.device,
        dtype=torch.float32,
    ) / 255.0
    corrected_tensor = corrected_tensor.to(dtype=sampled_bhwc.dtype)
    if int(sampled_bhwc.shape[-1]) > 3:
        corrected_tensor = torch.cat([corrected_tensor, sampled_bhwc[..., 3:]], dim=-1)
    if sampled_original_ndim == 3:
        corrected_tensor = corrected_tensor[0]
    info.update({
        "changed": True,
        "reason": last_reason,
        "accepted": accepted_count,
    })
    return corrected_tensor.clamp(0.0, 1.0), info


def apply_sift_drift_correction(reference_tile, sampled_tile, index=None, mode="SIFT"):
    mode = normalize_drift_correction_mode(mode)
    level = _drift_mode_level(mode)
    label = f"tile {index + 1}" if isinstance(index, int) else "standalone"
    info = {
        "changed": False,
        "reason": "disabled",
        "matches": 0,
        "inliers": 0,
        "mode": mode,
    }
    if mode == "Off":
        _drift_log(f"{label}: Off")
        return sampled_tile, info

    _drift_log(f"{label}: mode={mode} start")
    corrected, sift_info = _apply_sift_affine_correction(reference_tile, sampled_tile, index=index, enhanced=level >= 2, level=level)
    sift_info["mode"] = mode
    if level <= 2 or bool(sift_info.get("changed", False)):
        _drift_log(
            f"{label}: done changed={bool(sift_info.get('changed', False))} "
            f"stage=SIFT reason={sift_info.get('reason', 'unknown')} "
            f"matches={sift_info.get('matches', 0)} inliers={sift_info.get('inliers', 0)}"
        )
        return corrected, sift_info

    _drift_log(
        f"{label}: SIFT fallback triggered reason={sift_info.get('reason', 'unknown')} "
        f"matches={sift_info.get('matches', 0)} inliers={sift_info.get('inliers', 0)}"
    )
    flow_corrected, flow_info = _apply_reverse_flow_fallback(reference_tile, sampled_tile, level=level)
    flow_info["mode"] = mode
    flow_info["sift_reason"] = sift_info.get("reason", "unknown")
    flow_info["matches"] = sift_info.get("matches", 0)
    flow_info["inliers"] = sift_info.get("inliers", 0)
    if bool(flow_info.get("changed", False)):
        _drift_log(f"{label}: done changed=True stage=fallback reason={flow_info.get('reason', 'unknown')}")
        return flow_corrected, flow_info
    flow_info["reason"] = f"{flow_info.get('reason', 'flow_failed')} after sift:{sift_info.get('reason', 'unknown')}"
    _drift_log(f"{label}: done changed=False reason={flow_info.get('reason', 'unknown')}")
    return sampled_tile, flow_info
