"""Convert the G1 MJCF to USD with the standalone Isaac Sim MJCF importer (no Kit)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supermimic.config import load_config

settings = load_config()

from pxr import Plug

here = os.path.dirname(os.path.abspath(__file__))
import isaacsim.asset.importer.utils.impl as _iu

Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(_iu.__file__), "schemas"))
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

src = os.path.join(settings["robot_description_dir"], "g1_29dof.xml")
dst = settings["robot_usd_dir"]
cfg = MJCFImporterConfig(mjcf_path=os.path.normpath(src), usd_path=os.path.normpath(dst))
print(MJCFImporter(cfg).import_mjcf())
