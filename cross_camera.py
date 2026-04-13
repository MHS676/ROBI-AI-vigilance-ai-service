"""
cross_camera.py — Cross-Camera Layout Registry
================================================

Defines named "layouts" that describe how multiple cameras cover the same
set of service desks from different physical angles.

Each layout contains:
  • A primary_camera_id  — the frontal / highest-resolution camera
  • One CameraRoiLayout per physical camera, each holding a list of DeskRoi
    entries that map a desk_id to a pixel bounding box on that camera's frame

The layouts are registered in a module-level dict (LAYOUT_REGISTRY) at import
time.  The NestJS backend or a Super Admin can register custom layouts via
POST /layout; the defaults here cover the standard Falcon branch configuration.

Built-in layout  "layout_cam1_cam2"
─────────────────────────────────────
  Desk layout (top-down view, 640×480 frames assumed):

       [Desk 1 — Teller A]   [Desk 2 — Teller B]   [Desk 3 — Teller C]
                ↑ service counters facing customers ↑

  Camera 1 — frontal wide-angle (captures all 3 desks, side view of faces):
       Desk 1  x=  0, y=0, w=213, h=480   (left third)
       Desk 2  x=213, y=0, w=214, h=480   (centre third)
       Desk 3  x=427, y=0, w=213, h=480   (right third)

  Camera 2 — side angle from the left (captures customer faces head-on):
       Desk 1  x=400, y=0, w=240, h=480   (customers at Desk 1 face right)
       Desk 2  x=200, y=0, w=240, h=480   (customers at Desk 2 face slightly right)
       Desk 3  x=  0, y=0, w=250, h=480   (customers at Desk 3 face left / away)

  When a face is detected in Camera 1's Desk 1 zone AND in Camera 2's Desk 1
  zone, the BestViewSelector picks the view with the higher clarity_score for
  the final emotion analysis (Objective 13 — Irate Customer).

Usage:
    from cross_camera import get_layout, register_layout, LAYOUT_REGISTRY
"""

from __future__ import annotations

from schemas import BoundingBox, CameraRoiLayout, CrossCameraLayout, DeskRoi

# ─────────────────────────────────────────────────────────────────────────────
# Camera 1 — frontal wide-angle
# Covers all 3 desks; frame is horizontally divided into equal thirds.
# Best for general YOLO detection (weapon, fight, fall).
# Faces are viewed from the side — lower clarity for emotion analysis.
# ─────────────────────────────────────────────────────────────────────────────

_CAM1_DESK_ROIS: list[DeskRoi] = [
    DeskRoi(
        desk_id="desk_1",
        desk_label="Desk 1 — Teller A",
        table_id=None,                               # set by NestJS at provision time
        bounding_box=BoundingBox(x=0,   y=0, w=213, h=480),
    ),
    DeskRoi(
        desk_id="desk_2",
        desk_label="Desk 2 — Teller B",
        table_id=None,
        bounding_box=BoundingBox(x=213, y=0, w=214, h=480),
    ),
    DeskRoi(
        desk_id="desk_3",
        desk_label="Desk 3 — Teller C",
        table_id=None,
        bounding_box=BoundingBox(x=427, y=0, w=213, h=480),
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Camera 2 — side angle (left side of the counter row)
# Captures customer faces head-on or three-quarter view.
# Best for emotion / expression analysis (higher face clarity scores).
# Desk positions are mirrored / shifted compared to Camera 1.
# ─────────────────────────────────────────────────────────────────────────────

_CAM2_DESK_ROIS: list[DeskRoi] = [
    DeskRoi(
        desk_id="desk_1",
        desk_label="Desk 1 — Teller A",
        table_id=None,
        bounding_box=BoundingBox(x=400, y=0, w=240, h=480),
    ),
    DeskRoi(
        desk_id="desk_2",
        desk_label="Desk 2 — Teller B",
        table_id=None,
        bounding_box=BoundingBox(x=200, y=0, w=240, h=480),
    ),
    DeskRoi(
        desk_id="desk_3",
        desk_label="Desk 3 — Teller C",
        table_id=None,
        bounding_box=BoundingBox(x=0,   y=0, w=250, h=480),
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Built-in layouts
# ─────────────────────────────────────────────────────────────────────────────

_BUILTIN_LAYOUTS: list[CrossCameraLayout] = [
    CrossCameraLayout(
        layout_id="layout_cam1_cam2",
        description=(
            "Standard Falcon branch — Camera 1 (frontal wide-angle, 3 desk thirds) + "
            "Camera 2 (side-angle face view, mirrored desk positions). "
            "Covers 3 teller desks. Used for Objective 13: Irate Customer."
        ),
        primary_camera_id="camera_1",
        cameras=[
            CameraRoiLayout(camera_id="camera_1", desk_rois=_CAM1_DESK_ROIS),
            CameraRoiLayout(camera_id="camera_2", desk_rois=_CAM2_DESK_ROIS),
        ],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# In-memory registry  (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

LAYOUT_REGISTRY: dict[str, CrossCameraLayout] = {
    layout.layout_id: layout for layout in _BUILTIN_LAYOUTS
}


def get_layout(layout_id: str) -> CrossCameraLayout | None:
    """Look up a registered layout by ID.  Returns None if not found."""
    return LAYOUT_REGISTRY.get(layout_id)


def register_layout(layout: CrossCameraLayout) -> None:
    """
    Register (or overwrite) a layout in the in-memory registry.

    Called by POST /layout.  Layouts are NOT persisted to disk; they reset
    on service restart.  For permanent custom layouts, add them to
    _BUILTIN_LAYOUTS above.
    """
    LAYOUT_REGISTRY[layout.layout_id] = layout


def update_desk_table_id(
    layout_id: str,
    camera_id: str,
    desk_id: str,
    table_id: str,
) -> bool:
    """
    Link a DB table_id to a specific desk in an existing layout.

    Called by NestJS when a Table is provisioned and its table_id becomes
    known.  Mutates the layout in-place.

    Returns True if the desk was found and updated, False otherwise.
    """
    layout = LAYOUT_REGISTRY.get(layout_id)
    if not layout:
        return False
    cam_layout = layout.get_camera_layout(camera_id)
    if not cam_layout:
        return False
    roi = cam_layout.get_roi_for_desk(desk_id)
    if not roi:
        return False
    roi.table_id = table_id
    return True
