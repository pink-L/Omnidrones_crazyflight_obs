# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os

import torch
from isaacsim import SimulationApp
from tensordict import TensorDict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def init_simulation_app(cfg):
    # launch the simulator
    config = {"headless": cfg["headless"], "anti_aliasing": 1}
    # [2026-09-08] 本地 WebRTC livestream: +enable_livestream=true [+livestream_port=N]
    #   启用 omni.kit.livestream.webrtc -> 本机浏览器访问 https://<server-ip>:<port>/streaming
    #   (需要 GUI/headless=false 提供 viewport 供推流; 默认关闭不影响现有用法)
    if cfg.get("enable_livestream", False):
        _ls_port = int(cfg.get("livestream_port", 8211))
        config["extra_args"] = [
            "--enable", "omni.kit.livestream.webrtc",
            "--/app/livestream/protocol=webrtc",
            f"--/app/livestream/port={_ls_port}",
        ]
        _addr = cfg.get("livestream_address", None)
        if _addr:
            config["extra_args"].append(f"--/app/livestream/publicEndpointAddress={_addr}")
        print(f"[init_simulation_app] livestream enabled (webrtc, port={_ls_port}, "
              f"url=https://<server-ip>:{_ls_port}/streaming)", flush=True)
    # Isaac Sim 5.1: use base.kit for GUI (includes viewport), base.python.kit for headless
    import isaacsim as _isaacsim
    _isaacsim_path = os.path.dirname(_isaacsim.__file__)
    if cfg["headless"]:
        _exp = os.path.join(_isaacsim_path, "apps", "isaacsim.exp.base.python.kit")
    else:
        _exp = os.path.join(_isaacsim_path, "apps", "isaacsim.exp.base.kit")
    simulation_app = SimulationApp(config, experience=_exp)
    return simulation_app


def _get_shapes(self: TensorDict):
    return {
        k: v.shape if isinstance(v, torch.Tensor) else v.shapes for k, v in self.items()
    }


def _get_devices(self: TensorDict):
    return {
        k: v.device if isinstance(v, torch.Tensor) else v.devices
        for k, v in self.items()
    }


TensorDict.shapes = property(_get_shapes)
TensorDict.devices = property(_get_devices)
