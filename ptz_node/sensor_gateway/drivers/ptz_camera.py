"""PTZ camera driver — bridges to the ptz-agent camera + detector stack.

Backend (sim panorama vs. Reolink hardware) is decided entirely inside
``ptz-agent`` via env/state, so this driver never talks to hardware directly;
it only calls the ptz-agent facade after :func:`bootstrap_ptz_agent_runtime`
puts that project on ``sys.path``.
"""

from __future__ import annotations

from typing import Any

from ptz_node.bootstrap import bootstrap_ptz_agent_runtime
from ptz_node.sensor_gateway.base import BaseDriver, DeviceInfo, DriverError

_PTZ_CAPS = [
    "get_position",
    "move_to",
    "pan_by",
    "tilt_by",
    "set_fov_h",
    "snapshot",
    "detect",
    "caption",
]


class PTZCameraDriver(BaseDriver):
    kind = "ptz_camera"
    interface = "network"
    read_only = False

    def __init__(self, device_id: str, *, ptz_agent_root: str | None = None) -> None:
        super().__init__(device_id)
        self._ptz_agent_root = ptz_agent_root
        self._boot_path = None

    # -- ptz-agent bridge --------------------------------------------------

    def _ensure(self):
        if self._boot_path is None:
            self._boot_path = bootstrap_ptz_agent_runtime(self._ptz_agent_root)
        return self._boot_path

    def _camera(self):
        self._ensure()
        from tools import ptz_facade

        ptz_facade.warm_reolink_worker()
        return ptz_facade.get_ptz_camera()

    def _backend(self) -> str:
        self._ensure()
        from tools import ptz_facade

        return ptz_facade.ptz_backend_name()

    # -- contract ----------------------------------------------------------

    def describe(self) -> DeviceInfo:
        backend = ""
        try:
            backend = self._backend()
        except Exception:
            backend = "unavailable"
        return DeviceInfo(
            id=self.device_id,
            kind=self.kind,
            interface=self.interface,
            backend=backend,
            description="Pan/tilt/zoom camera with on-edge vision (YOLO/BioCLIP/Gemma4).",
            read_only=False,
            capabilities=list(_PTZ_CAPS),
            paths={"ptz_agent_project": str(self._boot_path or "")},
        )

    def invoke(self, capability: str, **params: Any) -> dict[str, Any]:
        fn = getattr(self, f"_cap_{capability}", None)
        if fn is None:
            raise DriverError(
                f"unknown capability {capability!r} for {self.device_id}; "
                f"choose from {_PTZ_CAPS}"
            )
        return fn(**params)

    # -- capabilities ------------------------------------------------------

    def _cap_get_position(self, **_: Any) -> dict[str, Any]:
        return dict(self._camera().get_position())

    def _cap_move_to(self, pan: float = 0.0, tilt: float = 0.0, **_: Any) -> dict[str, Any]:
        return dict(self._camera().move_to(float(pan), float(tilt)))

    def _cap_pan_by(self, degrees: float = 0.0, **_: Any) -> dict[str, Any]:
        return dict(self._camera().pan_by(float(degrees)))

    def _cap_tilt_by(self, degrees: float = 0.0, **_: Any) -> dict[str, Any]:
        return dict(self._camera().tilt_by(float(degrees)))

    def _cap_set_fov_h(self, fov_h: float = 60.0, **_: Any) -> dict[str, Any]:
        return dict(self._camera().set_fov_h(float(fov_h)))

    def _cap_snapshot(self, filename: str | None = None, **_: Any) -> dict[str, Any]:
        return {"path": self._camera().snapshot(filename=filename)}

    def _cap_detect(
        self,
        model: str = "yolo",
        targets: str = "*",
        target_taxon: str = "",
        target: str = "",
        max_soft_tokens: int | None = None,
        tile: bool = False,
        tile_size: int | None = None,
        tile_overlap: int = 0,
        tile_iou: float = 0.45,
        **_: Any,
    ) -> dict[str, Any]:
        cam = self._camera()
        viewport = cam._crop_viewport()
        self._ensure()
        from tools.detectors import detect

        # Sliced batch inference: tile a large viewport into model-sized crops.
        tile_kw: dict[str, Any] = {}
        if tile:
            tile_kw["tile"] = True
            if tile_size is not None:
                tile_kw["tile_size"] = int(tile_size)
            tile_kw["tile_overlap"] = int(tile_overlap)
            tile_kw["tile_iou"] = float(tile_iou)

        model_l = str(model).lower()
        if model_l in ("gemma4",):
            hint = (target or "").strip() or (
                targets if str(targets).strip() not in ("", "*") else ""
            )
            gkw: dict[str, Any] = {"target": hint}
            if max_soft_tokens is not None:
                gkw["max_soft_tokens"] = int(max_soft_tokens)
            return detect(viewport, model="gemma4", **gkw, **tile_kw)
        return detect(viewport, model=model_l, targets=targets,
                      target_taxon=target_taxon, **tile_kw)

    def _cap_caption(
        self,
        model: str = "bioclip",
        prompt: str = "",
        max_soft_tokens: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        cam = self._camera()
        viewport = cam._crop_viewport()
        self._ensure()
        from tools.detectors import caption

        ckw: dict[str, Any] = {}
        if str(model).lower() == "gemma4":
            if prompt:
                ckw["prompt"] = str(prompt)
            if max_soft_tokens is not None:
                ckw["max_soft_tokens"] = int(max_soft_tokens)
        return caption(viewport, model=model, **ckw)

    # -- detector availability (node-level, not per-capability) ------------

    def detector_status(self) -> dict[str, Any]:
        self._ensure()
        from tools.detectors import available_models

        return {"models": available_models()}
