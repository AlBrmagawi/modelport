"""Include the built dashboard in distributable wheels and source archives."""

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class DashboardBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version == "editable":
            return
        dashboard = Path(self.root) / "web" / "dist"
        if not (dashboard / "index.html").is_file():
            if os.environ.get("MODELPORT_REQUIRE_WEB") == "1":
                raise RuntimeError(
                    "Build the dashboard with npm ci and npm run build before release"
                )
            return
        build_data["force_include"][str(dashboard)] = (
            "modelport/web" if self.target_name == "wheel" else "web/dist"
        )
