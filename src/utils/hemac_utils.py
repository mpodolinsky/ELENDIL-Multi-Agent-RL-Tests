import yaml
from hemac import HeMAC_v0

def make_parallel_env(hemac_config, render_mode=None):
        with open(hemac_config, "r") as f:
            cfg = yaml.safe_load(f)

        # Override render_mode if provided
        if cfg["render_mode"] is None and render_mode is not None:
            cfg["render_mode"] = render_mode

        env = HeMAC_v0.parallel_env(**cfg)
        return env