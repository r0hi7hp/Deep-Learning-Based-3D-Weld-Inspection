import json, sys, math, copy
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from tqdm import tqdm


###############################################################################
# 1. Parameters – tune these once and keep them together
###############################################################################
SOURCE_PLY   = "scan_with_defects.ply"   # the component containing flaws
TARGET_PLY   = "reference_nominal.ply"   # the flawless / CAD / golden part
VOXEL_SIZE   = 0.25                      # ↓ increase ⇢ smoother, ↑ faster
ICP_MAX_DIST = VOXEL_SIZE * 2
DEFECT_TH    = VOXEL_SIZE * 1.5          # points farther than this are “defect”
DB_EPS       = VOXEL_SIZE * 2
ROI_MULT     = 2.0                      # multiplier for very high positive deviation ROI
DB_MIN_PTS   = 50                       # stricter clustering to avoid noise
CLUSTER_ISO_TH = VOXEL_SIZE * 15        # clusters whose centroid is farther than this from any other are discarded
ISO_LEVEL    = "B"                       # ISO 5817 quality level (B/C/D)
JSON_OUT     = "defect_report.json"


###############################################################################
# 2. Utility functions
###############################################################################
def preprocess(pcd, voxel_size):
    pcd_d = pcd.voxel_down_sample(voxel_size)            # ↓ sparse but regular
    pcd_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                           radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
             pcd_d,
             o3d.geometry.KDTreeSearchParamHybrid(
                   radius=voxel_size * 5, max_nn=100))
    return pcd_d, fpfh


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load a PLY file and always return an Open3D PointCloud. If the PLY stores
    a mesh, its vertices are uniformly sampled into a point cloud."""
    pcd = o3d.io.read_point_cloud(path)
    # If the file is a mesh or the resulting cloud is very sparse, switch to mesh
    if len(pcd.points) < 5000:
        mesh = o3d.io.read_triangle_mesh(path)
        if not mesh.has_vertices():
            raise ValueError(f"'{path}' does not contain any vertices")
        # Sample a dense cloud (≈50k pts) for robust registration
        pcd = mesh.sample_points_uniformly(number_of_points=max(50_000, len(mesh.vertices)))
    return pcd


def execute_fast_global_reg(src_down, tgt_down, f_src, f_tgt, voxel):
    distance = voxel * 1.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src_down, tgt_down, f_src, f_tgt,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance))
    return result

def ransac_fpfh_registration(src, tgt, voxel):
    """Coarse alignment using RANSAC on down-sampled FPFH features."""
    radius_normal = voxel * 2
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel * 5
    f_src = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    f_tgt = o3d.pipelines.registration.compute_fpfh_feature(
        tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    dist_th = voxel * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, f_src, f_tgt, True, dist_th,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_th)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_icp(src, tgt, initial_trans, max_dist):
    return o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, initial_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

def pointwise_distance(src, tgt):
    """
    For every point in src compute distance to nearest neighbour in tgt.
    Returns np.array of distances, shape = (N,).
    """
    tgt_kd = o3d.geometry.KDTreeFlann(tgt)
    dist   = np.zeros(len(src.points))
    for i, p in enumerate(src.points):
        _, idx, d2 = tgt_kd.search_knn_vector_3d(p, 1)
        dist[i] = math.sqrt(d2[0])
    return dist

def bbox_numpy(points):
    mn, mx = points.min(0), points.max(0)
    return mn.tolist(), mx.tolist()

def pca_principal_axis(points_np: np.ndarray):
    mean = points_np.mean(axis=0)
    X = points_np - mean
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    axis = vecs[:, 0]
    ortho_std = math.sqrt(max(vals[1], 1e-12))
    elong_ratio = (vals[0] / max(vals[1], 1e-12)) if vals[1] > 0 else float('inf')
    return mean, axis, ortho_std, elong_ratio

def select_points_in_axis_band(all_points_np: np.ndarray, center: np.ndarray, axis: np.ndarray,
                               radius: float, t_min: float, t_max: float):
    v = all_points_np - center
    t = v @ axis
    proj = np.outer(t, axis)
    radial = v - proj
    d = np.linalg.norm(radial, axis=1)
    mask = (d <= radius) & (t >= t_min) & (t <= t_max)
    return np.where(mask)[0]

def segment_top_planes(pcd: o3d.geometry.PointCloud, voxel: float, max_planes: int = 2):
    ds = pcd.voxel_down_sample(max(voxel, 1e-6))
    ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    remaining = ds
    planes = []
    for _ in range(max_planes):
        if len(remaining.points) < 1000:
            break
        model, inliers = remaining.segment_plane(distance_threshold=voxel * 1.5,
                                                 ransac_n=3, num_iterations=1000)
        if len(inliers) < 500:
            break
        a, b, c, d = model
        n_raw = np.array([a, b, c], dtype=float)
        nrm = np.linalg.norm(n_raw) + 1e-12
        n = n_raw / nrm
        d = float(d) / nrm
        pts = np.asarray(remaining.select_by_index(inliers).points)
        planes.append((n, float(d), pts))
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes

def intersect_planes(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float):
    # Planes: n·x + d = 0
    v = np.cross(n1, n2)
    nv2 = np.dot(v, v)
    if nv2 < 1e-10:
        return None, None
    # A point on the line from formula using cross products
    p = (np.cross(v, n2) * (-d1) + np.cross(n1, v) * (-d2)) / nv2
    v = v / np.linalg.norm(v)
    return p, v

###############################################################################
# 3. MAIN PIPELINE
###############################################################################
def main():
    ##### Load ################################################################
    print("Loading...")
    src_raw = load_point_cloud(SOURCE_PLY)
    tgt_raw = load_point_cloud(TARGET_PLY)
    print(f"Source points: {np.asarray(src_raw.points).shape[0]}")
    print(f"Target points: {np.asarray(tgt_raw.points).shape[0]}")

    ##### Pre-processing ######################################################
    print("Down-sampling, estimating normals & FPFH...")
    src_down, f_src = preprocess(src_raw, VOXEL_SIZE)
    tgt_down, f_tgt = preprocess(tgt_raw, VOXEL_SIZE)

    ##### RANSAC-FPFH – coarse alignment #####################################
    print("RANSAC + FPFH (coarse) and FGR fallback...")
    src_coarse = src_raw.voxel_down_sample(VOXEL_SIZE)
    tgt_coarse = tgt_raw.voxel_down_sample(VOXEL_SIZE)
    result_ransac = ransac_fpfh_registration(src_coarse, tgt_coarse, VOXEL_SIZE)
    result_fgr = execute_fast_global_reg(src_down, tgt_down, f_src, f_tgt, VOXEL_SIZE)
    best_result = result_ransac if result_ransac.fitness >= result_fgr.fitness else result_fgr
    print(best_result)
    src_fgr_aligned = copy.deepcopy(src_raw).transform(best_result.transformation)

    ##### Ensure normals for ICP (point-to-plane needs normals) ###############
    src_fgr_aligned.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=VOXEL_SIZE * 2, max_nn=30))
    tgt_raw.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=VOXEL_SIZE * 2, max_nn=30))

    ##### ICP – fine alignment ###############################################
    print("ICP (fine)...")
    result_icp = refine_icp(src_fgr_aligned, tgt_raw, np.eye(4), ICP_MAX_DIST)
    print(result_icp)
    src_aligned = copy.deepcopy(src_raw).transform(result_icp.transformation)

    ##### Distance field #####################################################
    print("Computing point-wise distances...")
    # Compute signed deviations (positive & negative)
    # Positive deviation – excess material on the scan vs reference
    dist_pos = pointwise_distance(src_aligned, tgt_raw)
    # Negative deviation – missing material (reference protrudes beyond scan)
    dist_neg = pointwise_distance(tgt_raw, src_aligned)

    # Masks for basic defect threshold
    mask_pos = dist_pos > DEFECT_TH
    mask_neg = dist_neg > DEFECT_TH

    # ROI – points where POSITIVE deviation is very high (excess material only)
    roi_pos = dist_pos > (DEFECT_TH * ROI_MULT)
    pts_roi_pos = src_aligned.select_by_index(np.where(roi_pos)[0])
    defects_pcd = pts_roi_pos if len(pts_roi_pos.points) > 0 else None

    if defects_pcd is not None:
        print(f"Restricting clustering to ROI: {len(defects_pcd.points)} pts with deviation > {ROI_MULT}×th (positive only)")
    else:
        print("No high-positive-deviation ROI points found, falling back to basic positive deviation set.")
        defects_pcd = src_aligned.select_by_index(np.where(mask_pos)[0])

    total_defect_pts = len(defects_pcd.points)
    print(f"Defect candidate points: {total_defect_pts}")

    # Prefer seam from target planes (intersection line). Fallback to PCA band.
    used_plane_band = False
    seam_ctx = {"p0": None, "axis": None, "band_radius": None, "n1": None, "d1": None, "n2": None, "d2": None}
    try:
        planes = segment_top_planes(tgt_raw, VOXEL_SIZE, max_planes=2)
        if len(planes) >= 2:
            n1, d1, pts1 = planes[0]
            n2, d2, pts2 = planes[1]
            ang = math.degrees(math.acos(np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)))
            if 10.0 < ang < 170.0:
                p0, axis = intersect_planes(n1, d1, n2, d2)
                if p0 is not None:
                    # Determine seam extent from target points near both planes
                    tgt_xyz = np.asarray(tgt_raw.points)
                    v_all = tgt_xyz - p0
                    t_vals = v_all @ axis
                    # keep central 98% to ignore outliers
                    t_min = np.percentile(t_vals, 1)
                    t_max = np.percentile(t_vals, 99)
                    # band radius from how tightly target points concentrate around line
                    radial = np.linalg.norm(v_all - np.outer(t_vals, axis), axis=1)
                    band_radius = max(DEFECT_TH, np.percentile(radial, 80))
                    all_xyz = np.asarray(src_aligned.points)
                    idx_band = select_points_in_axis_band(all_xyz, p0, axis, band_radius,
                                                          t_min - VOXEL_SIZE * 2, t_max + VOXEL_SIZE * 2)
                    print(f"Plane-seam band points: {len(idx_band)} (radius≈{band_radius:.3f}, ang≈{ang:.1f}°)")
                    if len(idx_band) > 0:
                        weld_band = src_aligned.select_by_index(idx_band)
                        idx_pos = np.where(dist_pos[idx_band] > DEFECT_TH)[0]
                        defects_pcd = weld_band.select_by_index(idx_pos)
                        total_defect_pts = len(defects_pcd.points)
                        print(f"Defect candidates within plane-seam band: {total_defect_pts}")
                        used_plane_band = True
                        seam_ctx = {"p0": p0, "axis": axis, "band_radius": float(band_radius),
                                    "n1": n1, "d1": float(d1), "n2": n2, "d2": float(d2)}
    except Exception as e:
        print(f"Plane-based seam detection failed: {e}")

    if not used_plane_band and total_defect_pts > 0:
        # PCA fallback
        roi_xyz = np.asarray(defects_pcd.points)
        center, axis, ortho_std, elong = pca_principal_axis(roi_xyz)
        if elong > 5.0:
            v = roi_xyz - center
            t_vals = v @ axis
            t_min, t_max = np.min(t_vals), np.max(t_vals)
            band_radius = max(DEFECT_TH, 2.5 * ortho_std)
            all_xyz = np.asarray(src_aligned.points)
            idx_band = select_points_in_axis_band(all_xyz, center, axis, band_radius,
                                                  t_min - VOXEL_SIZE * 2, t_max + VOXEL_SIZE * 2)
            print(f"Weld band points: {len(idx_band)} (radius≈{band_radius:.3f}, elong={elong:.2f})")
            if len(idx_band) > 0:
                weld_band = src_aligned.select_by_index(idx_band)
                idx_pos = np.where(dist_pos[idx_band] > DEFECT_TH)[0]
                defects_pcd = weld_band.select_by_index(idx_pos)
                total_defect_pts = len(defects_pcd.points)
                print(f"Defect candidates within weld band: {total_defect_pts}")
                seam_ctx = {"p0": center, "axis": axis, "band_radius": float(band_radius),
                            "n1": None, "d1": None, "n2": None, "d2": None}

    ##### Clustering with DBSCAN #############################################
    print("DBSCAN clustering...")
    if total_defect_pts == 0:
        print("No points exceeded the defect threshold – finished.")
        return

    xyz = np.asarray(defects_pcd.points)
    db  = DBSCAN(eps=DB_EPS, min_samples=DB_MIN_PTS).fit(xyz)
    labels = db.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {num_clusters} clusters")

    ##### Map clusters -> ISO 5817 defect type ################################
    # NOTE: ISO 5817 groups imperfections mainly by length (l), height (h),
    # projection beyond surface, and sometimes area. Below are **very** rough
    # heuristics; refine them to your needs & to the specific welding process.
    def classify_iso5817(points):
        mn, mx = points.min(0), points.max(0)
        l = np.linalg.norm(mx - mn)          # crude length
        h = (mx[2] - mn[2])                  # assume Z is “height/outward”
        if h > 2.0 * VOXEL_SIZE:
            return "Excess Weld Metal (ISO 5817 501)"
        elif l > 10 * VOXEL_SIZE:
            return "Undercut (ISO 5817 401)"
        elif len(points) < 200:
            return "Porosity (ISO 5817 130)"
        else:
            return "Irregular profile (ISO 5817 504)"

    raw_clusters = []
    for lbl in range(num_clusters):
        pts = xyz[labels == lbl]
        if pts.shape[0] == 0:
            continue
        centroid = pts.mean(0).tolist()
        bb_min, bb_max = bbox_numpy(pts)
        raw_clusters.append({
            "label": int(lbl),
            "num_points": int(pts.shape[0]),
            "centroid": centroid,
            "bbox_min": bb_min,
            "bbox_max": bb_max,
            "defect_type": classify_iso5817(pts)
        })

    # Drop clusters away from seam line and planes
    if seam_ctx["axis"] is not None and seam_ctx["p0"] is not None and len(raw_clusters) > 0:
        p0 = seam_ctx["p0"]; axis = seam_ctx["axis"]; br = seam_ctx["band_radius"] or DEFECT_TH
        n1 = seam_ctx["n1"]; d1 = seam_ctx["d1"]; n2 = seam_ctx["n2"]; d2 = seam_ctx["d2"]
        keep = []
        margin = VOXEL_SIZE * 1.5
        plane_th = max(VOXEL_SIZE * 2.0, br * 0.6)
        for c in raw_clusters:
            cpt = np.array(c["centroid"])  # centroid in source aligned space
            v = cpt - p0
            t = np.dot(v, axis)
            d_line = np.linalg.norm(v - t * axis)
            ok_line = d_line <= (br + margin)
            ok_planes = True
            if n1 is not None and n2 is not None:
                dpl1 = abs(np.dot(n1, cpt) + d1)
                dpl2 = abs(np.dot(n2, cpt) + d2)
                ok_planes = (dpl1 <= plane_th) and (dpl2 <= plane_th)
            if ok_line and ok_planes:
                keep.append(c)
        print(f"Clusters kept after seam filtering: {len(keep)}/{len(raw_clusters)}")
        raw_clusters = keep

    # ------------------------------------------------------------------
    #  Isolated cluster removal: keep clusters whose centroid is close
    #  to at least one other centroid within CLUSTER_ISO_TH.
    # ------------------------------------------------------------------
    report = []
    centroids_np = np.array([c["centroid"] for c in raw_clusters])
    for i, cinfo in enumerate(raw_clusters):
        if len(raw_clusters) == 1:
            break  # single cluster, keep it
        dists = np.linalg.norm(centroids_np - centroids_np[i], axis=1)
        dists = np.delete(dists, i)  # exclude self
        if dists.size and dists.min() <= CLUSTER_ISO_TH:
            report.append({k: v for k, v in cinfo.items() if k != "label"})
    print(f"Clusters kept after isolation filter: {len(report)}/{len(raw_clusters)}")

    ##### Save JSON ###########################################################
    with open(JSON_OUT, "w") as f:
        json.dump(report, f, indent=4)
    print(f"JSON saved → {JSON_OUT}")

    ##### Optional – visualize ###############################################
    # Color weld area red, rest of source green
    src_colored = copy.deepcopy(src_aligned)
    src_colors = np.tile(np.array([[0.1, 0.8, 0.1]]), (len(src_colored.points), 1))
    if seam_ctx.get("p0") is not None and seam_ctx.get("axis") is not None:
        if seam_ctx.get("band_radius") is not None:
            p0 = seam_ctx["p0"]; axis = seam_ctx["axis"]; br = seam_ctx["band_radius"]
            xyz_all = np.asarray(src_aligned.points)
            v = xyz_all - p0
            t = v @ axis
            radial = v - np.outer(t, axis)
            d = np.linalg.norm(radial, axis=1)
            in_band = d <= (br + VOXEL_SIZE * 0.5)
            src_colors[in_band] = np.array([0.9, 0.0, 0.0])
    src_colored.colors = o3d.utility.Vector3dVector(src_colors)
    o3d.visualization.draw_geometries([
        tgt_raw.paint_uniform_color([0.7,0.7,0.7]),
        src_colored
    ])
    # o3d.io.write_point_cloud("colored_source.ply", src_colored)

###############################################################################
# 4. Helper to give each cluster a colour
###############################################################################
def plt_colormap(labels, alpha=1.0):
    """
    Map each label to a colour – small helper using matplotlib’s tab20 palette.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    n = max(labels) + 1
    cmap = cm.get_cmap('tab20', n if n > 0 else 1)
    colors = np.array([cmap(l) for l in labels])
    colors[:,3] = alpha
    return colors[:,:3]


if __name__ == "__main__":
    main()
