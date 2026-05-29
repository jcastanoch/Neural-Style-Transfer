"""Reconstrucciones por capa al estilo de la Figura 1 de Gatys et al. (CVPR 2016).

Visualiza QUE informacion conserva cada capa de una CNN reconstruyendo una
imagen a partir de la respuesta de esa capa. Dos experimentos comparten la figura:

    Reconstrucciones de CONTENIDO (fila inferior, paneles a-e)
        Se parte de ruido blanco y se optimizan los pixeles para que las
        activaciones de UNA sola capa coincidan con las de la imagen de
        contenido (conv1_2 .. conv5_2). Las capas tempranas reconstruyen la
        foto casi perfecta; las profundas conservan el contenido de alto nivel
        pero descartan el pixel exacto.

    Reconstrucciones de ESTILO (fila superior, paneles a-e)
        Misma idea, pero igualando las *matrices de Gram* (correlaciones entre
        canales = representacion de estilo) sobre subconjuntos CRECIENTES de
        capas: a = {conv1_1}, b = {conv1_1, conv2_1}, ... e = {conv1_1..conv5_1}.

Por que la version ingenua daba ruido (y como se corrige aqui):
    * Sin un prior de variacion total (TV) la inversion de capas profundas
      esta muy subdeterminada y el optimizador rellena su espacio nulo con
      "speckle" multicolor en vez de imagenes suaves y reconocibles
      (Mahendran & Vedaldi, 2015). -> se anade TV a las de contenido.
    * El paper usa average pooling (Seccion 2); el max pooling da
      reconstrucciones mas duras. -> se reemplaza max por avg.
    * La perdida de estilo con la normalizacion 1/(4 N^2 M^2) es minuscula y
      L-BFGS apenas se mueve. -> se escala (style_scale) para condicionar bien.
    * El wrapper Vgg19 del repo solo expone 6 capas y no llega a conv*_2 /
      conv5_2. -> se usa un extractor VGG-19 flexible (cualquier conv, post-ReLU).

Uso:
    python reconstruction_figure.py --content c1.jpg --style s1.jpg
    python reconstruction_figure.py --content c1.jpg --style s1.jpg \
        --image_size 384 --steps 300        # alta calidad (lento en CPU)

Salida: data/output-images/reconstructions_<contenido>_<estilo>.png
Probado en CPU (fallback automatico) y CUDA.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# torchvision pretrained VGG espera entradas RGB en [0,1] normalizadas con estas stats.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONTENT_LAYERS = ("conv1_2", "conv2_2", "conv3_2", "conv4_2", "conv5_2")
STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
STYLE_SUBSETS = [STYLE_LAYERS[: k + 1] for k in range(len(STYLE_LAYERS))]
PANEL_LETTERS = "abcde"

STYLE_COLOR = "#d95f0e"     # calido  -> fila de estilo
CONTENT_COLOR = "#2c7fb8"   # frio    -> fila de contenido
FRAME_COLOR = "#cccccc"


# ----------------------------------------------------------------------------
# Extractor VGG-19 flexible (cualquier capa conv con nombre, post-ReLU, avg pool)
# ----------------------------------------------------------------------------

def build_vgg(device: torch.device):
    """Construye una vez la lista de modulos VGG-19 (avg pooling) y el mapa
    nombre_de_capa -> indice del modulo cuya salida (post-ReLU) la representa."""
    raw = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    layers: list[nn.Module] = []
    name_to_index: dict[str, int] = {}
    block, conv_in_block, pending = 1, 0, None

    for layer in raw:
        if isinstance(layer, nn.Conv2d):
            conv_in_block += 1
            pending = f"conv{block}_{conv_in_block}"
            layers.append(layer)
        elif isinstance(layer, nn.ReLU):
            layers.append(nn.ReLU(inplace=False))
            name_to_index[pending] = len(layers) - 1   # capturamos tras la ReLU
        elif isinstance(layer, nn.MaxPool2d):
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            block += 1
            conv_in_block = 0
        else:
            layers.append(layer)

    seq = nn.Sequential(*layers).to(device).eval()
    for p in seq.parameters():
        p.requires_grad_(False)
    return seq, name_to_index


class Extractor:
    """Vista (truncada) de la VGG compartida que devuelve activaciones de las
    capas pedidas. Se trunca al indice mas profundo necesario para no calcular
    trabajo de mas (clave en CPU para los paneles superficiales)."""

    def __init__(self, seq, name_to_index, needed: Iterable[str], mean, std):
        self.seq = seq
        self.capture = {name_to_index[n]: n for n in needed}
        self.max_idx = max(self.capture)
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        out: dict[str, torch.Tensor] = {}
        for i in range(self.max_idx + 1):
            x = self.seq[i](x)
            name = self.capture.get(i)
            if name is not None:
                out[name] = x
        return out


# ----------------------------------------------------------------------------
# Perdidas
# ----------------------------------------------------------------------------

def gram_matrix(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    b, ch, h, w = x.size()
    features = x.view(b, ch, h * w)
    gram = features.bmm(features.transpose(1, 2))
    if normalize:
        gram = gram / (ch * h * w)
    return gram


def style_loss_per_layer(feat: torch.Tensor, target_gram: torch.Tensor) -> torch.Tensor:
    return ((gram_matrix(feat) - target_gram) ** 2).sum()


def content_match(F: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """MSE de activaciones normalizado por la energia del objetivo, para que un
    unico peso de TV se comporte igual en capas de magnitudes muy distintas."""
    return ((F - P) ** 2).sum() / (P ** 2).sum().clamp_min(1e-8)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


# ----------------------------------------------------------------------------
# Reconstruccion por optimizacion (ruido blanco -> iguala una representacion)
# ----------------------------------------------------------------------------

def _white_noise(like: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    # randn con shape explicito -> tensor contiguo (randn_like heredaria los
    # strides no contiguos del permute y L-BFGS falla al hacer .view(-1) del grad).
    noise = torch.randn(tuple(like.shape), device=like.device, dtype=like.dtype)
    img = (noise * 0.1 + 0.5).clamp(0.0, 1.0).contiguous()
    return img.requires_grad_(True)


def _run_lbfgs(x: torch.Tensor, steps: int, loss_fn) -> torch.Tensor:
    optimizer = optim.LBFGS([x], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    run = [0]
    last = [0.0]
    while run[0] <= steps:
        def closure() -> torch.Tensor:
            with torch.no_grad():
                x.clamp_(0.0, 1.0)
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            run[0] += 1
            last[0] = float(loss.detach())
            return loss
        optimizer.step(closure)
    with torch.no_grad():
        x.clamp_(0.0, 1.0)
    return x.detach(), last[0]


def reconstruct_content(seq, nti, mean, std, content_img, layer, steps, seed, tv_weight):
    model = Extractor(seq, nti, [layer], mean, std)
    with torch.no_grad():
        target = model(content_img)[layer].detach()
    x = _white_noise(content_img, seed)

    def loss_fn():
        return content_match(model(x)[layer], target) + tv_weight * total_variation(x)

    return _run_lbfgs(x, steps, loss_fn)


def reconstruct_style(seq, nti, mean, std, style_img, layers, steps, seed, style_scale):
    model = Extractor(seq, nti, layers, mean, std)
    with torch.no_grad():
        targets = {l: gram_matrix(model(style_img)[l]).detach() for l in layers}
    w = 1.0 / len(layers)
    x = _white_noise(style_img, seed)

    def loss_fn():
        feats = model(x)
        per_layer = sum(style_loss_per_layer(feats[l], targets[l]) for l in layers)
        return style_scale * w * per_layer

    return _run_lbfgs(x, steps, loss_fn)


# ----------------------------------------------------------------------------
# Figura (estilo Figura 1 del paper)
# ----------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor):
    return t.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()


def _show(ax, img, title=None, subtitle=None, title_color="#222", bold_frame=False):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(title_color if bold_frame else FRAME_COLOR)
        s.set_linewidth(1.6 if bold_frame else 1.0)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=6)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8.5, color="#555", family="monospace")


def _style_subtitle(subset) -> str:
    if len(subset) == 1:
        return subset[0]
    return f"{subset[0]} ... {subset[-1]}"


def make_figure(style_img, content_img, style_recons, content_recons, save_path):
    fig = plt.figure(figsize=(14.5, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    subfigs = fig.subfigures(2, 1, hspace=0.04)

    sf_style = subfigs[0]
    sf_style.suptitle(
        "Reconstrucciones de ESTILO  -  matrices de Gram sobre subconjuntos crecientes de capas",
        fontsize=13, fontweight="bold", color=STYLE_COLOR,
    )
    axes = sf_style.subplots(1, 6)
    _show(axes[0], _to_numpy(style_img), title="Imagen de estilo",
          title_color=STYLE_COLOR, bold_frame=True)
    for i, (subset, recon) in enumerate(style_recons):
        _show(axes[i + 1], _to_numpy(recon),
              title=f"({PANEL_LETTERS[i]})", subtitle=_style_subtitle(subset),
              title_color=STYLE_COLOR)

    sf_content = subfigs[1]
    sf_content.suptitle(
        "Reconstrucciones de CONTENIDO  -  activaciones de una sola capa",
        fontsize=13, fontweight="bold", color=CONTENT_COLOR,
    )
    axes = sf_content.subplots(1, 6)
    _show(axes[0], _to_numpy(content_img), title="Imagen de contenido",
          title_color=CONTENT_COLOR, bold_frame=True)
    for i, (layer, recon) in enumerate(content_recons):
        _show(axes[i + 1], _to_numpy(recon),
              title=f"({PANEL_LETTERS[i]})", subtitle=layer,
              title_color=CONTENT_COLOR)

    fig.suptitle(
        "Reconstruccion de representaciones en una CNN  (Gatys et al., CVPR 2016 - Fig. 1)",
        fontsize=15.5, fontweight="bold",
    )
    fig.supxlabel(
        "Estilo (arriba): al sumar capas profundas (a->e) las estadisticas de Gram capturan textura a escalas "
        "crecientes, descartando la disposicion global de la escena.\n"
        "Contenido (abajo): las capas tempranas (a) reconstruyen los pixeles casi perfecto; las profundas (e) "
        "conservan el contenido de alto nivel pero pierden el detalle exacto.",
        fontsize=10, style="italic", color="#333",
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# IO / CLI
# ----------------------------------------------------------------------------

def resolve_path(arg: str, default_dir: str) -> str:
    if os.path.exists(arg):
        return arg
    candidate = os.path.join(default_dir, arg)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f'No se encontro la imagen: {arg}')


def load_image(path: str, max_size: int, device: torch.device) -> torch.Tensor:
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f'cv2 no pudo leer: {path}')
    img = img[:, :, ::-1]                                   # BGR -> RGB
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv.INTER_AREA if scale < 1 else cv.INTER_CUBIC
    img = cv.resize(np.ascontiguousarray(img), (new_w, new_h), interpolation=interp)
    t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0).to(device)


def save_panel(t: torch.Tensor, path: str):
    img = (_to_numpy(t) * 255.0).astype("uint8")
    cv.imwrite(path, img[:, :, ::-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstrucciones de contenido y estilo (Gatys et al. 2016, Fig. 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--content", default="c1.jpg", help="imagen de contenido (nombre en data/content-images o ruta)")
    p.add_argument("--style", default="s1.jpg", help="imagen de estilo (nombre en data/style-images o ruta)")
    p.add_argument("--output", default=None, help="ruta del PNG combinado (por defecto en data/output-images)")
    p.add_argument("--image_size", type=int, default=320, help="lado mayor de la imagen de trabajo (menor = mas rapido)")
    p.add_argument("--steps", type=int, default=250, help="evaluaciones de L-BFGS por reconstruccion")
    p.add_argument("--style_scale", type=float, default=1e6, help="multiplicador de condicionamiento para la (diminuta) perdida de estilo")
    p.add_argument("--tv_weight", type=float, default=0.6, help="suavizado por variacion total en las reconstrucciones de contenido (0.6 limpia las capas profundas sin borronear las superficiales)")
    p.add_argument("--save_panels", action="store_true", help="guardar tambien cada reconstruccion como PNG individual")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    if device.type == "cuda":
        print(f"[info] gpu   : {torch.cuda.get_device_name(0)}")

    here = Path(__file__).resolve().parent
    content_path = resolve_path(args.content, str(here / "data" / "content-images"))
    style_path = resolve_path(args.style, str(here / "data" / "style-images"))

    content_img = load_image(content_path, args.image_size, device)
    style_img = load_image(style_path, args.image_size, device)
    print(f"[info] content {tuple(content_img.shape)}  style {tuple(style_img.shape)}")

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    seq, nti = build_vgg(device)

    content_label = Path(args.content).stem
    style_label = Path(args.style).stem
    out_path = args.output or str(here / "data" / "output-images" / f"reconstructions_{content_label}_{style_label}.png")
    panel_dir = Path(out_path).with_suffix("")
    if args.save_panels:
        panel_dir.mkdir(parents=True, exist_ok=True)

    style_recons = []
    for i, subset in enumerate(STYLE_SUBSETS):
        print(f"[style {i + 1}/{len(STYLE_SUBSETS)}] {_style_subtitle(subset)} ...", flush=True)
        recon, loss = reconstruct_style(seq, nti, mean, std, style_img, subset, args.steps, args.seed, args.style_scale)
        print(f"          loss={loss:.4e}")
        style_recons.append((subset, recon))
        if args.save_panels:
            save_panel(recon, str(panel_dir / f"style_{PANEL_LETTERS[i]}.png"))

    content_recons = []
    for i, layer in enumerate(CONTENT_LAYERS):
        print(f"[content {i + 1}/{len(CONTENT_LAYERS)}] {layer} ...", flush=True)
        recon, loss = reconstruct_content(seq, nti, mean, std, content_img, layer, args.steps, args.seed, args.tv_weight)
        print(f"          loss={loss:.4e}")
        content_recons.append((layer, recon))
        if args.save_panels:
            save_panel(recon, str(panel_dir / f"content_{PANEL_LETTERS[i]}.png"))

    make_figure(style_img, content_img, style_recons, content_recons, out_path)
    print(f"[info] listo -> {out_path}")
    if args.save_panels:
        print(f"[info] paneles -> {panel_dir}/")


if __name__ == "__main__":
    main()
