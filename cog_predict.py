import os
import importlib.util
import json
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from pathlib import Path

import cog


dirs = {
    "art": "good_art_1k_512",
    "faces": "good_ffhq_full_512",
    "dog": "trial_dog",
    "panda": "trial_panda",
    "shell": "trial_shell",
    "skull": "trial_skull",
}


class Model(cog.Model):
    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.noise_dim = 256
        self.nets = {}
        self.im_sizes = {}
        for key in dirs:
            self.nets[key], self.im_sizes[key] = self.load_model(key)

    @cog.input(
        "variant",
        type=str,
        default="art",
        options=sorted(dirs.keys()),
        help="Model variant",
    )
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(self, variant, seed):
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        torch.manual_seed(seed)

        batch = 1
        dist = "/tmp/output"
        os.makedirs(dist, exist_ok=True)
        out_path = os.path.join(dist, "generated.png")

        net_ig = self.nets[variant]
        im_size = self.im_sizes[variant]

        with torch.no_grad():
            noise = torch.randn(batch, self.noise_dim).to(self.device)
            g_imgs = net_ig(noise)[0]
            g_imgs = F.interpolate(g_imgs, im_size)
            g_img = g_imgs[0]
            img = g_img.add(1).mul(0.5)
            vutils.save_image(img, out_path)
        return Path(out_path)

    def load_model(self, key):
        model_dir = os.path.join("models", dirs[key])
        conf_path = os.path.join(model_dir, "args.txt")
        with open(conf_path) as f:
            conf = json.load(f)

        iteration = conf["iter"]
        im_size = conf["im_size"]

        spec = importlib.util.spec_from_file_location(
            "generator", os.path.join(model_dir, "models.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        net_ig = mod.Generator(ngf=64, nz=self.noise_dim, nc=3, im_size=im_size)
        net_ig.to(self.device)

        ckpt = os.path.join(model_dir, "models", f"{iteration}.pth")
        checkpoint = torch.load(ckpt, map_location=lambda a, b: a)
        net_ig.load_state_dict(checkpoint["g"])

        net_ig.to(self.device)

        return net_ig, im_size
