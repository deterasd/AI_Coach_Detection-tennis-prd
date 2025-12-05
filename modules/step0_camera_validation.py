"""
Step 0: Camera Matrix & 3D/2D Consistency Validation
====================================================

目的：
  在執行 Step1 重投影誤差分析之前，先檢查：

  1. 雙目標定產生的相機矩陣是否合理
     - 投影矩陣 P1 / P2 結構
     - 內參矩陣 K（fx, fy, cx, cy, skew）
     - 旋轉矩陣 R 是否接近正交、det(R) 是否接近 +1
     - baseline（兩相機中心距離）是否接近預期（例如約 400mm）

  2. 3D JSON 軌跡與 2D JSON 軌跡是否與 P1 / P2 相容
     - 使用 P1 將 3D 點投影到 side 2D
     - 使用 P2 將 3D 點投影到 45° 2D
     - 計算投影誤差的統計量（mean/median/max）
     - 檢查是否存在明顯座標系/尺度不一致（例如誤差上百 px）

  3. 統一輸出一份 JSON 結果，用於後續 Step1 參考
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datetime import datetime

# --------------------------------------------------------
# 基本工具函式（完全自帶，不依賴 utils.py）
# --------------------------------------------------------


def load_json_file(path: str) -> Any:
    """載入 JSON 檔案並回傳內容。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON 檔案不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_results(data: dict, path: str) -> None:
    """將結果儲存為 JSON，處理 numpy 型別為可序列化格式。"""

    def _convert(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=_convert)


def generate_output_path(base_input: str, suffix: str) -> str:
    """
    根據輸入檔案產生輸出檔名：
      例如 input: foo/bar/xxx.json, suffix: '_step0_camera_validation'
      -> foo/bar/xxx_step0_camera_validation.json
    """
    base_dir = os.path.dirname(base_input)
    base_name = os.path.basename(base_input)
    name, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".json"
    new_name = f"{name}{suffix}{ext}"
    return os.path.join(base_dir, new_name)


# --------------------------------------------------------
# 線性代數 / 投影相關
# --------------------------------------------------------


def rq_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """執行 3x3 矩陣的 RQ 分解以取得 K 與 R。"""
    if matrix.shape != (3, 3):
        raise ValueError("RQ 分解僅支援 3x3 矩陣")

    m = matrix.astype(float)
    # 利用 QR 分解推導 RQ，避免依賴額外套件
    q, r = np.linalg.qr(np.flipud(m).T)
    r = np.flipud(r.T)
    q = q.T
    r = np.fliplr(r)
    q = np.flipud(q)

    diag = np.sign(np.diag(r))
    diag[diag == 0] = 1
    d = np.diag(diag)
    r = r @ d
    q = d @ q

    return r, q  # 這裡 r 對應 K, q 對應 R


def extract_intrinsics_from_P(P: np.ndarray) -> Optional[np.ndarray]:
    """從投影矩陣 P 中提取 3x3 內參矩陣 K，若失敗則返回 None。"""
    try:
        if P.shape != (3, 4):
            return None
        M = P[:, :3]
        if np.linalg.matrix_rank(M) < 3:
            return None
        K, _ = rq_decomposition(M)
        if abs(K[2, 2]) > 1e-8:
            K = K / K[2, 2]
        return K
    except Exception:
        return None


def compute_camera_center(P: np.ndarray) -> Optional[np.ndarray]:
    """
    從投影矩陣 P 計算相機中心 C（世界座標）
    公式：C = - M^{-1} * p4
    其中 P = [M | p4]，M 為 3x3，p4 為第 4 欄。
    """
    try:
        if P.shape != (3, 4):
            return None
        M = P[:, :3]
        p4 = P[:, 3]
        if np.linalg.matrix_rank(M) < 3:
            return None
        Minv = np.linalg.inv(M)
        C = -Minv @ p4
        return C  # shape (3,)
    except Exception:
        return None


def project_point(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    使用投影矩陣 P 將 3D 齊次座標 X (4,) 投影到 2D。

    回傳:
        np.ndarray: [x, y] 若 w 接近 0 則回傳 [nan, nan]
    """
    x = P @ X
    if abs(x[2]) < 1e-8:
        return np.array([np.nan, np.nan], dtype=float)
    return x[:2] / x[2]


def safe_ratio(numerator: float, denominator: float) -> float:
    """避免零除的比例計算，回傳百分比數值。"""
    return float(numerator / denominator * 100) if denominator > 0 else 0.0


# --------------------------------------------------------
# 相機矩陣驗證
# --------------------------------------------------------


def validate_camera_intrinsics(
    P: np.ndarray,
    name: str,
    ref_K: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    驗證投影矩陣 P 的內參合理性，並與 calib.json 中的 K 做比較（若提供）。

    回傳:
        dict 包含：
          - is_valid
          - intrinsics: fx, fy, cx, cy, skew, aspect_ratio
          - ref_intrinsics (如果 ref_K 有提供)
          - warnings, errors
    """
    result: Dict[str, Any] = {
        "camera_name": name,
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "intrinsics": {},
        "ref_intrinsics": None,
    }

    if P.shape != (3, 4):
        result["errors"].append(f"P 形狀錯誤，預期 (3,4)，實際 {P.shape}")
        result["is_valid"] = False
        return result

    K = extract_intrinsics_from_P(P)
    if K is None:
        result["errors"].append("無法從 P 分解出有效的內參矩陣 K，請檢查 P 是否為合法投影矩陣")
        result["is_valid"] = False
        return result

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    skew = K[0, 1]

    if fx <= 0 or fy <= 0:
        result["errors"].append(f"焦距異常：fx={fx:.2f}, fy={fy:.2f}，應為正數")
        result["is_valid"] = False

    aspect_ratio = fx / fy if fy != 0 else 0.0
    if abs(aspect_ratio - 1.0) > 0.1:
        result["warnings"].append(f"焦距比例 fx/fy={aspect_ratio:.3f} 偏離 1，請確認像素比例是否非方形")

    if cx < 0 or cy < 0:
        result["warnings"].append(f"主點座標 cx={cx:.2f}, cy={cy:.2f} 有負值，請確認影像座標系")

    if abs(skew) > 10:
        result["warnings"].append(f"skew={skew:.2f} 絕對值較大，通常應接近 0，請確認輸入")

    result["intrinsics"] = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "skew": float(skew),
        "aspect_ratio": float(aspect_ratio),
        "K_matrix": K,
    }

    # 若提供参照內參，做比較
    if ref_K is not None and ref_K.shape == (3, 3):
        r_fx, r_fy = ref_K[0, 0], ref_K[1, 1]
        r_cx, r_cy = ref_K[0, 2], ref_K[1, 2]
        r_skew = ref_K[0, 1]
        result["ref_intrinsics"] = {
            "fx": float(r_fx),
            "fy": float(r_fy),
            "cx": float(r_cx),
            "cy": float(r_cy),
            "skew": float(r_skew),
            "K_matrix": ref_K,
        }

        # 簡單比較差異
        df_fx = abs(fx - r_fx)
        df_fy = abs(fy - r_fy)
        df_cx = abs(cx - r_cx)
        df_cy = abs(cy - r_cy)
        df_skew = abs(skew - r_skew)

        if df_fx > 5 or df_fy > 5:
            result["warnings"].append(
                f"P 分解出的 fx, fy 與 calib.json 中差異偏大 "
                f"(Δfx={df_fx:.2f}, Δfy={df_fy:.2f})"
            )
        if df_cx > 10 or df_cy > 10:
            result["warnings"].append(
                f"P 分解出的 cx, cy 與 calib.json 中差異偏大 "
                f"(Δcx={df_cx:.2f}, Δcy={df_cy:.2f})"
            )
        if df_skew > 1:
            result["warnings"].append(
                f"P 分解出的 skew 與 calib.json 中差異較大 (Δskew={df_skew:.2f})"
            )

    return result


def validate_rotation_matrix(R: np.ndarray, name: str) -> Dict[str, Any]:
    """
    檢查旋轉矩陣 R 是否接近正交、det(R) 是否接近 +1。
    """
    result: Dict[str, Any] = {
        "name": name,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "det": None,
        "orthogonality_error": None,
    }

    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        result["errors"].append(f"R 形狀需為 (3,3)，目前為 {R.shape}")
        result["is_valid"] = False
        return result

    RtR = R.T @ R
    I = np.eye(3)
    ortho_err = float(np.linalg.norm(RtR - I, ord="fro"))
    det = float(np.linalg.det(R))

    result["det"] = det
    result["orthogonality_error"] = ortho_err

    if ortho_err > 1e-2:
        result["warnings"].append(
            f"R^T R 與 I 差距偏大，||R^T R - I||_F = {ortho_err:.4e}，"
            "請檢查是否為正確旋轉矩陣"
        )
    if abs(det - 1.0) > 1e-2:
        result["warnings"].append(
            f"det(R) = {det:.4f}，與 1 有明顯差距，可能非純旋轉"
        )

    if ortho_err > 5e-2 or abs(det - 1.0) > 5e-2:
        result["is_valid"] = False

    return result


def compute_baseline(C1: Optional[np.ndarray], C2: Optional[np.ndarray]) -> Optional[float]:
    """計算兩相機中心距離（mm）。"""
    if C1 is None or C2 is None:
        return None
    return float(np.linalg.norm(C1 - C2))


# --------------------------------------------------------
# 3D / 2D JSON 解析與一致性檢查
# --------------------------------------------------------


def infer_keypoints_from_frames(frames: List[dict]) -> List[str]:
    """
    從 3D 或 2D JSON 的 frame 列表中推斷 keypoint 名稱列表。
    排除:
      - frame
      - tennis_ball_hit
      - tennis_ball_angle
      - 其他非 dict 或缺 x/y/z 的欄位
    """
    keys = set()
    for f in frames:
        if not isinstance(f, dict):
            continue
        for k, v in f.items():
            if k in ("frame", "tennis_ball_hit", "tennis_ball_angle"):
                continue
            if not isinstance(v, dict):
                continue
            # 至少要有 x,y 或 x,y,z 其中之一
            if any(coord in v for coord in ("x", "y", "z")):
                keys.add(k)
    return sorted(keys)


def get_keypoint_3d(frame: dict, name: str) -> Optional[np.ndarray]:
    """安全取得 3D keypoint (x, y, z)；若不存在或有 None 則回傳 None。"""
    kp = frame.get(name)
    if not isinstance(kp, dict):
        return None
    x = kp.get("x")
    y = kp.get("y")
    z = kp.get("z")
    if x is None or y is None or z is None:
        return None
    return np.array([float(x), float(y), float(z)], dtype=float)


def get_keypoint_2d(frame: dict, name: str) -> Optional[np.ndarray]:
    """安全取得 2D keypoint (x, y)；若不存在或有 None 則回傳 None。"""
    kp = frame.get(name)
    if not isinstance(kp, dict):
        return None
    x = kp.get("x")
    y = kp.get("y")
    if x is None or y is None:
        return None
    return np.array([float(x), float(y)], dtype=float)


def ensure_frame_list(data: Any, name: str) -> List[dict]:
    """
    確保載入的 JSON 是 frame list 形式：
      - 若是 list -> 直接回傳
      - 若是 dict 且有 'frames' -> 回傳 data['frames']
      - 否則丟出錯誤
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
        return data["frames"]
    raise ValueError(f"{name} JSON 格式不符合預期，需為 list 或含 'frames' 的 dict")


def analyze_projection_consistency(
    frames_3d: List[dict],
    frames_2d_side: List[dict],
    frames_2d_45: List[dict],
    P1: np.ndarray,
    P2: np.ndarray,
    sample_max_frames: int = 300,
) -> Dict[str, Any]:
    """
    使用 P1/P2 將 3D 點投影回 2D，與 2D JSON 比較誤差。
    主要是檢查「3D world 是否真的與 P1/P2 在同一座標系統與尺度」。
    """

    num_frames = min(len(frames_3d), len(frames_2d_side), len(frames_2d_45))
    if num_frames == 0:
        raise ValueError("3D/2D JSON 幀數為 0，無法進行一致性檢查")

    # 抽樣 frame index（避免全部幀數太大）
    if num_frames <= sample_max_frames:
        frame_indices = list(range(num_frames))
    else:
        # 等距抽樣
        step = num_frames / float(sample_max_frames)
        frame_indices = sorted({int(i * step) for i in range(sample_max_frames)})

    # 推斷共同 keypoints 名稱（3D 與 2D 都存在的點）
    kp_3d = set(infer_keypoints_from_frames(frames_3d))
    kp_2d_side = set(infer_keypoints_from_frames(frames_2d_side))
    kp_2d_45 = set(infer_keypoints_from_frames(frames_2d_45))
    common_kp = sorted(kp_3d & kp_2d_side & kp_2d_45)

    if not common_kp:
        raise ValueError("3D / 2D JSON 間沒有共同 keypoint 名稱，無法對應")

    # 收集誤差
    errors_side: List[float] = []
    errors_45: List[float] = []
    sample_records: List[dict] = []

    for fi in frame_indices:
        f3d = frames_3d[fi]
        f2d_s = frames_2d_side[fi]
        f2d_d = frames_2d_45[fi]

        for kp in common_kp:
            X = get_keypoint_3d(f3d, kp)
            if X is None:
                continue
            x_side = get_keypoint_2d(f2d_s, kp)
            x_45 = get_keypoint_2d(f2d_d, kp)

            X_h = np.array([X[0], X[1], X[2], 1.0], dtype=float)

            if x_side is not None:
                proj_s = project_point(P1, X_h)
                if not np.any(np.isnan(proj_s)):
                    err_s = float(np.linalg.norm(x_side - proj_s))
                    errors_side.append(err_s)
                    sample_records.append(
                        {
                            "frame": fi,
                            "camera": "side",
                            "keypoint": kp,
                            "error": err_s,
                        }
                    )

            if x_45 is not None:
                proj_d = project_point(P2, X_h)
                if not np.any(np.isnan(proj_d)):
                    err_d = float(np.linalg.norm(x_45 - proj_d))
                    errors_45.append(err_d)
                    sample_records.append(
                        {
                            "frame": fi,
                            "camera": "45",
                            "keypoint": kp,
                            "error": err_d,
                        }
                    )

    errors_side_arr = np.array(errors_side, dtype=float) if errors_side else np.array([])
    errors_45_arr = np.array(errors_45, dtype=float) if errors_45 else np.array([])

    def stats(arr: np.ndarray) -> Dict[str, Any]:
        if arr.size == 0:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "max": None,
                "p95": None,
            }
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    summary_side = stats(errors_side_arr)
    summary_45 = stats(errors_45_arr)

    # 粗略評估：是否有明顯座標系/尺度錯誤（誤差全部超大）
    # 這裡用比較寬鬆的門檻：若所有誤差均 > 100px，則判定為「明顯不相容」
    def detect_incompatibility(arr: np.ndarray) -> bool:
        if arr.size == 0:
            return False
        return bool(np.all(arr > 100.0))

    incompatible_side = detect_incompatibility(errors_side_arr)
    incompatible_45 = detect_incompatibility(errors_45_arr)

    return {
        "num_frames_total": num_frames,
        "num_frames_sampled": len(frame_indices),
        "sampled_frame_indices": frame_indices,
        "common_keypoints": common_kp,
        "side_stats": summary_side,
        "cam45_stats": summary_45,
        "side_incompatible": incompatible_side,
        "cam45_incompatible": incompatible_45,
        "example_errors": sorted(sample_records, key=lambda r: r["error"], reverse=True)[
            :20
        ],
    }


# --------------------------------------------------------
# 主流程：Step 0 驗證
# --------------------------------------------------------


def run_step0_camera_validation(
    calib_json_path: str,
    json_3d_path: str,
    json_2d_side_path: str,
    json_2d_45_path: str,
    output_json_path: Optional[str] = None,
    expected_baseline_mm: float = 400.0,
) -> Dict[str, Any]:
    """
    Step 0 主流程：

    1. 讀取 calibration_results.json，取得：
       - 左相機/左前相機內參、畸變、R、T、投影矩陣
    2. 驗證 P1 / P2 內參合理性
    3. 驗證 R 的正交性 / det(R)
    4. 計算相機中心與 baseline
    5. 使用 3D / 2D JSON 進行投影一致性檢查
    """

    print("\n" + "=" * 80)
    print("Step 0: Camera Matrix & 3D/2D Consistency Validation")
    print("=" * 80)

    # 1) 讀 calibration_results.json
    print(f"\n[INFO] 讀取標定結果: {calib_json_path}")
    calib = load_json_file(calib_json_path)

    def _get_matrix(name: str) -> Optional[np.ndarray]:
        v = calib.get(name)
        if v is None:
            return None
        arr = np.array(v, dtype=float)
        return arr

    K1 = _get_matrix("左相機內參矩陣")
    K2 = _get_matrix("左前相機內參矩陣")
    D1 = _get_matrix("左相機畸變係數")
    D2 = _get_matrix("左前相機畸變係數")
    R = _get_matrix("旋轉矩陣")
    T = _get_matrix("平移向量")
    P1 = _get_matrix("左相機投影矩陣")
    P2 = _get_matrix("左前相機投影矩陣")

    if P1 is None or P2 is None:
        raise ValueError("calibration_results.json 中缺少『左相機投影矩陣』或『左前相機投影矩陣』")

    print("\n[INFO] 相機矩陣資訊：")
    print(f"  左相機投影矩陣 P1 形狀: {P1.shape}")
    print(f"  左前相機投影矩陣 P2 形狀: {P2.shape}")

    # 2) 驗證 P1 / P2 內參
    print("\n" + "-" * 80)
    print("[STEP] 驗證 P1 / P2 內參矩陣")
    print("-" * 80)

    intr_side = validate_camera_intrinsics(P1, "Side Camera", ref_K=K1)
    intr_45 = validate_camera_intrinsics(P2, "45° Camera", ref_K=K2)

    def _print_intrinsics(info: Dict[str, Any]) -> None:
        name = info["camera_name"]
        print(f"\n[{name}]")
        if not info["is_valid"]:
            print("  [!] 內參驗證結果：不通過")
        else:
            print("  [OK] 內參驗證結果：通過（基本結構合理）")
        intr = info["intrinsics"]
        print(
            f"  fx={intr.get('fx'):.2f}, fy={intr.get('fy'):.2f}, "
            f"cx={intr.get('cx'):.2f}, cy={intr.get('cy'):.2f}, skew={intr.get('skew'):.2f}"
        )
        print(f"  aspect_ratio = {intr.get('aspect_ratio'):.4f}")
        if info["warnings"]:
            print("  Warnings:")
            for w in info["warnings"]:
                print(f"    - {w}")
        if info["errors"]:
            print("  Errors:")
            for e in info["errors"]:
                print(f"    - {e}")
        if info.get("ref_intrinsics"):
            r = info["ref_intrinsics"]
            print(
                "  參考 calib.json 內參："
                f"fx={r['fx']:.2f}, fy={r['fy']:.2f}, cx={r['cx']:.2f}, cy={r['cy']:.2f}, skew={r['skew']:.2f}"
            )

    _print_intrinsics(intr_side)
    _print_intrinsics(intr_45)

    # 3) 驗證 R、計算 baseline
    print("\n" + "-" * 80)
    print("[STEP] 驗證旋轉矩陣 R 並計算雙目 baseline")
    print("-" * 80)

    rot_info = None
    if R is not None:
        rot_info = validate_rotation_matrix(R, "Stereo R")
        print("\n[Stereo R]")
        print(f"  det(R) = {rot_info['det']:.6f}")
        print(f"  ||R^T R - I||_F = {rot_info['orthogonality_error']:.4e}")
        if rot_info["warnings"]:
            print("  Warnings:")
            for w in rot_info["warnings"]:
                print(f"    - {w}")
        if rot_info["errors"]:
            print("  Errors:")
            for e in rot_info["errors"]:
                print(f"    - {e}")
        if rot_info["is_valid"]:
            print("  [OK] R 看起來像是合法的旋轉矩陣")
        else:
            print("  [!] R 可能不是合法的旋轉矩陣，請檢查 stereoCalibrate 結果")
    else:
        print("  [!] calibration_results.json 中找不到『旋轉矩陣』欄位，無法驗證 R")

    # 相機中心與 baseline（從 P1/P2）：
    C1 = compute_camera_center(P1)
    C2 = compute_camera_center(P2)
    baseline = compute_baseline(C1, C2)

    if C1 is not None and C2 is not None:
        print("\n[Camera Centers from P1/P2]")
        print(f"  C1 (Side Camera)  = {C1}")
        print(f"  C2 (45° Camera)   = {C2}")
        if baseline is not None:
            print(f"  baseline |C2 - C1| = {baseline:.3f} （單位與標定世界座標相同，預期約 {expected_baseline_mm:.1f} ）")
            diff_baseline = abs(baseline - expected_baseline_mm)
            if diff_baseline > 50.0:
                print(
                    f"  [!] baseline 與預期 {expected_baseline_mm:.1f} mm 差異較大 (Δ={diff_baseline:.1f}mm)，"
                    "請檢查標定尺度是否正確（square_size、單位等）"
                )
            else:
                print("  [OK] baseline 與預期相符，尺度看起來合理")
    else:
        print("\n[!] 無法從 P1/P2 計算相機中心，可能是 P 矩陣不可逆或格式有誤")

    # 4) 讀 3D / 2D JSON，做投影一致性檢查
    print("\n" + "-" * 80)
    print("[STEP] 讀取 3D / 2D JSON 並執行投影一致性檢查")
    print("-" * 80)

    print(f"[INFO] 讀取 3D JSON:      {json_3d_path}")
    print(f"[INFO] 讀取 2D Side JSON: {json_2d_side_path}")
    print(f"[INFO] 讀取 2D 45° JSON:  {json_2d_45_path}")

    data_3d_raw = load_json_file(json_3d_path)
    data_2d_side_raw = load_json_file(json_2d_side_path)
    data_2d_45_raw = load_json_file(json_2d_45_path)

    frames_3d = ensure_frame_list(data_3d_raw, "3D")
    frames_2d_side = ensure_frame_list(data_2d_side_raw, "2D Side")
    frames_2d_45 = ensure_frame_list(data_2d_45_raw, "2D 45°")

    print(
        f"\n[INFO] 幀數：3D={len(frames_3d)}, Side={len(frames_2d_side)}, 45°={len(frames_2d_45)}"
    )

    consistency = analyze_projection_consistency(
        frames_3d,
        frames_2d_side,
        frames_2d_45,
        P1,
        P2,
        sample_max_frames=300,
    )

    print("\n[RESULT] 3D → 2D 重投影誤差統計（從抽樣幀）")
    print("  共用 keypoints:", ", ".join(consistency["common_keypoints"]))
    print(
        f"  抽樣幀數: {consistency['num_frames_sampled']} / {consistency['num_frames_total']} "
        f"(frame indices 例：{consistency['sampled_frame_indices'][:5]} ...)"
    )

    side_stats = consistency["side_stats"]
    cam45_stats = consistency["cam45_stats"]

    def _print_stats(name: str, st: Dict[str, Any], incompatible: bool) -> None:
        print(f"\n  [{name}]")
        if st["count"] == 0:
            print("    無投影樣本（可能沒有共同 keypoints 或座標無效）")
            return
        print(
            f"    samples = {st['count']}, mean = {st['mean']:.2f}px, "
            f"median = {st['median']:.2f}px, max = {st['max']:.2f}px, p95 = {st['p95']:.2f}px"
        )
        if incompatible:
            print(
                "    [!] 所有誤差都大於 100px，極可能是：\n"
                "        - 3D world 與 P1/P2 的座標系不一致\n"
                "        - 或 3D 座標被放大/縮小（尺度錯誤）\n"
                "        - 或 3D 未使用這組 P1/P2 進行三角化"
            )
        else:
            # 簡單分級
            mean_err = st["mean"]
            if mean_err <= 5:
                level = "Excellent"
            elif mean_err <= 10:
                level = "Good"
            elif mean_err <= 30:
                level = "Fair"
            else:
                level = "Poor"
            print(f"    粗略品質等級：{level}")

    _print_stats("Side Camera", side_stats, consistency["side_incompatible"])
    _print_stats("45° Camera", cam45_stats, consistency["cam45_incompatible"])

    # 簡單建議
    print("\n" + "-" * 80)
    print("[STEP] 總結與建議")
    print("-" * 80)

    recommendations: List[str] = []

    if consistency["side_incompatible"] or consistency["cam45_incompatible"]:
        recommendations.append(
            "3D world 與 P1/P2 的投影誤差全部大於 100px，"
            "幾乎可以確定：目前使用的 3D 座標並不是用這組 P1/P2 做三角化。"
        )
        recommendations.append(
            "請確認：\n"
            "  1. 你在建立 3D 的時候，使用的相機內參/外參是否與 calib.json 一致\n"
            "  2. 是否有額外的世界座標轉換（旋轉/平移/尺度），未反映到 P1/P2 中\n"
            "  3. Step1 中硬編碼的 P1/P2 是否已改成這次標定輸出的投影矩陣"
        )
    else:
        # 誤差數值合理時的建議
        if side_stats["count"] > 0 and side_stats["mean"] is not None:
            m = side_stats["mean"]
            if m <= 5:
                recommendations.append(
                    "Side 相機 3D→2D 重投影誤差平均小於 5px，表示 3D 與 P1 在同一座標系且品質佳。"
                )
            elif m <= 15:
                recommendations.append(
                    "Side 相機 3D→2D 重投影誤差在 5~15px 之間，勉強可用，但可以考慮再優化標定或濾波。"
                )
            else:
                recommendations.append(
                    "Side 相機 3D→2D 重投影誤差大於 15px，建議檢查標定過程與座標對齊。"
                )

        if cam45_stats["count"] > 0 and cam45_stats["mean"] is not None:
            m = cam45_stats["mean"]
            if m <= 5:
                recommendations.append(
                    "45° 相機 3D→2D 重投影誤差平均小於 5px，表示 3D 與 P2 在同一座標系且品質佳。"
                )
            elif m <= 15:
                recommendations.append(
                    "45° 相機 3D→2D 重投影誤差在 5~15px 之間，勉強可用，但可以考慮再優化標定或濾波。"
                )
            else:
                recommendations.append(
                    "45° 相機 3D→2D 重投影誤差大於 15px，建議檢查標定過程與座標對齊。"
                )

    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec}")

    # 組合輸出結果
    results: Dict[str, Any] = {
        "metadata": {
            "analysis_time": datetime.now().isoformat(),
            "calibration_file": calib_json_path,
            "json_3d": json_3d_path,
            "json_2d_side": json_2d_side_path,
            "json_2d_45": json_2d_45_path,
            "expected_baseline_mm": expected_baseline_mm,
            "step": "Step0_Camera_Validation",
        },
        "camera_matrices": {
            "P1_side": P1,
            "P2_45": P2,
            "K1_side_from_calib": K1,
            "K2_45_from_calib": K2,
            "D1_side_from_calib": D1,
            "D2_45_from_calib": D2,
            "R_stereo": R,
            "T_stereo": T,
        },
        "intrinsics_validation": {
            "side": intr_side,
            "cam45": intr_45,
        },
        "rotation_validation": rot_info,
        "camera_centers": {
            "C1_side": C1,
            "C2_45": C2,
            "baseline_mm": baseline,
        },
        "projection_consistency": consistency,
        "recommendations": recommendations,
    }

    # 儲存 JSON
    if output_json_path is None:
        output_json_path = generate_output_path(json_3d_path, "_step0_camera_validation")

    save_json_results(results, output_json_path)
    print(f"\n[OK] Step0 結果已儲存至: {output_json_path}")

    return results


# --------------------------------------------------------
# CLI 入口
# --------------------------------------------------------

if __name__ == "__main__":

    # 預設範例（如未從 CLI 傳入參數）
    default_calib = "output/calibration_results.json"
    default_3d = "0306_3__trajectory/trajectory__2/0306_3__2(3D_trajectory_smoothed).json"
    default_2d_side = "0306_3__trajectory/trajectory__2/0306_3__2_side(2D_trajectory_smoothed).json"
    default_2d_45 = "0306_3__trajectory/trajectory__2/0306_3__2_45(2D_trajectory_smoothed).json"

    # 命令列參數支援（與 Step1 完全一致）
    if len(sys.argv) >= 5:
        calib_json_path = sys.argv[1]
        json_3d_path = sys.argv[2]
        json_2d_side_path = sys.argv[3]
        json_2d_45_path = sys.argv[4]
        output_json_path = sys.argv[5] if len(sys.argv) > 5 else None
        expected_baseline_mm = float(sys.argv[6]) if len(sys.argv) > 6 else 400.0
    else:
        # 使用預設路徑
        calib_json_path = default_calib
        json_3d_path = default_3d
        json_2d_side_path = default_2d_side
        json_2d_45_path = default_2d_45
        output_json_path = None
        expected_baseline_mm = 400.0

        print("提示：你也可以用命令列呼叫 Step0：")
        print("python step0_camera_validation.py <calib_json> <3d_json> <2d_side_json> <2d_45_json> [output_json] [expected_baseline_mm]")

    # 執行 Step0
    try:
        run_step0_camera_validation(
            calib_json_path,
            json_3d_path,
            json_2d_side_path,
            json_2d_45_path,
            output_json_path,
            expected_baseline_mm,
        )
    except Exception as e:
        print(f"\n[ERROR] Step0 驗證失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
