"""Galeria de resultados de NST: UN PNG por imagen de CONTENIDO ("familia").

Escanea data/output-images/ (las carpetas combined_<contenido>_<estilo>/),
agrupa por imagen de contenido y, por CADA contenido, arma un PNG presentable
("una familia"): la imagen original mas TODOS los estilos que se le aplicaron,
cada uno etiquetado con el nombre del estilo, en una cuadricula ordenada.

Los PNG salen en una carpeta nueva, AL MISMO NIVEL que content-images/ y
style-images/:

    data/familias/familia_<contenido>.png

Robusto a proposito (los resultados son historicos y desordenados):
  * Una carpeta combined_* sin .jpg dentro = pareja aun en proceso -> se salta.
  * Si el contenido ya no tiene su imagen original (p.ej. se renombro), igual
    arma la familia con sus estilos y marca "(sin original)".
  * Estilos que ya no estan en style-images/ (p.ej. estiloImpresionista) tambien
    se incluyen: la etiqueta sale del nombre de la carpeta, no del archivo.
  * Lee con PIL (soporta rutas con acentos/'n~' y formatos jpg/png/webp).

IMPORTANTE: correlo CUANDO el batch ya termino, asi cada familia sale completa
y todo queda bien clasificado.

Uso:
    python gallery_by_content.py
    python gallery_by_content.py --cols 5 --thumb 512 --dpi 140 --cell 3.0
    python gallery_by_content.py --out-dir data/familias

Salida por defecto: data/familias/  (un familia_<contenido>.png por contenido)
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "data" / "output-images"
CONTENT_DIR = HERE / "data" / "content-images"
STYLE_DIR = HERE / "data" / "style-images"
FAMILIES_DIR = HERE / "data" / "familias"   # <- nueva carpeta, hermana de las otras

ACCENT = "#2c7fb8"        # azul para el contenido / encabezados
FRAME_ORIG = "#2c7fb8"    # borde de la imagen original
FRAME_RESULT = "#cccccc"  # borde de los resultados
BG = "white"

try:                      # Pillow >= 9.1 usa Image.Resampling; antes Image.LANCZOS
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


# ----------------------------------------------------------------------------
# Descubrimiento y parseo de resultados
# ----------------------------------------------------------------------------

def known_stems(folder: Path) -> list[str]:
    """Stems (nombre sin extension) de los archivos de una carpeta, del mas
    largo al mas corto (para desambiguar prefijos al parsear)."""
    if not folder.is_dir():
        return []
    return sorted({p.stem for p in folder.iterdir() if p.is_file()}, key=len, reverse=True)


def parse_combined(folder_name: str, contents: list[str], styles: list[str]):
    """'combined_<contenido>_<estilo>' -> (contenido, estilo) o None.
    Usa los stems conocidos para cortar bien aunque haya '_' raros; si no,
    cae a partir por el ultimo '_'."""
    if not folder_name.startswith("combined_"):
        return None
    rest = folder_name[len("combined_"):]
    for c in contents:                                  # contenido conocido como prefijo
        if rest.startswith(c + "_"):
            return c, rest[len(c) + 1:]
    for s in styles:                                    # estilo conocido como sufijo
        if rest.endswith("_" + s):
            return rest[: -len(s) - 1], s
    c, sep, s = rest.rpartition("_")                    # fallback
    return (c, s) if sep else None


def pick_final_image(folder: Path, cstem: str, sstem: str):
    """El jpg/png final dentro de la carpeta. Prefiere el nombre cannonico de
    NST (<contenido>_<estilo>.jpg); si no, el mas reciente. None si no hay."""
    imgs = [p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not imgs:
        return None
    canonical = next((p for p in imgs if p.stem == f"{cstem}_{sstem}"), None)
    return canonical or max(imgs, key=lambda p: p.stat().st_mtime)


def gather(output_dir: Path, contents: list[str], styles: list[str]):
    """Devuelve (groups, skipped, in_progress).
    groups: {contenido -> [(estilo, ruta_resultado), ...]} ordenado por estilo."""
    groups: dict[str, list] = defaultdict(list)
    skipped, in_progress = [], []
    for d in sorted(output_dir.glob("combined_*")):
        if not d.is_dir():
            continue
        parsed = parse_combined(d.name, contents, styles)
        if parsed is None:
            skipped.append(d.name)
            continue
        cstem, sstem = parsed
        final = pick_final_image(d, cstem, sstem)
        if final is None:
            in_progress.append(d.name)
            continue
        groups[cstem].append((sstem, final))
    for c in groups:
        groups[c].sort(key=lambda t: t[0].lower())
    return groups, skipped, in_progress


# ----------------------------------------------------------------------------
# Carga de imagenes y etiquetas
# ----------------------------------------------------------------------------

def load_rgb(path, thumb: int):
    if path is None:
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    if thumb and max(img.size) > thumb:
        img.thumbnail((thumb, thumb), RESAMPLE)
    return np.asarray(img)


def pretty_content(stem: str) -> str:
    s = re.sub(r"^contenido", "", stem)
    return s or stem


def pretty_style(stem: str) -> str:
    s = re.sub(r"^estilo", "", stem)
    return s or stem


def _frame(ax, color, lw):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


# ----------------------------------------------------------------------------
# Una figura ("familia") por contenido
# ----------------------------------------------------------------------------

def build_familia(cstem, items, content_files, out_dir: Path,
                  thumb: int, dpi: int, cell: float, cols: int) -> Path:
    """Arma el PNG de UN contenido: original + todos sus estilos en cuadricula.

    La cuadricula 'envuelve' a 'cols' columnas para que el PNG quede balanceado
    aunque el contenido tenga muchos (o pocos) estilos.
    """
    n_total = 1 + len(items)                       # original + estilos
    ncols = max(1, min(cols, n_total))
    nrows = math.ceil(n_total / ncols)

    fig = plt.figure(figsize=(ncols * cell, nrows * cell + 0.9),
                     facecolor=BG, layout="constrained")
    mark = "" if cstem in content_files else "   (sin original)"
    fig.suptitle(f"{pretty_content(cstem)}    -    {len(items)} estilos{mark}",
                 fontsize=15, fontweight="bold", color=ACCENT)

    axes = fig.subplots(nrows, ncols, squeeze=False)
    flat = [ax for row in axes for ax in row]

    # --- celda 0: la imagen de contenido original ---
    orig = load_rgb(content_files.get(cstem), thumb)
    if orig is not None:
        flat[0].imshow(orig)
        flat[0].set_xlabel("ORIGINAL", fontsize=9.5, color=ACCENT, fontweight="bold")
    else:
        flat[0].imshow(np.full((8, 8, 3), 245, dtype="uint8"))
        flat[0].set_xlabel("(sin original)", fontsize=8.5, color="#999")
    _frame(flat[0], FRAME_ORIG, 2.5)

    # --- celdas 1..n: cada estilo aplicado ---
    for k, (sstem, path) in enumerate(items, start=1):
        img = load_rgb(path, thumb)
        if img is not None:
            flat[k].imshow(img)
        else:
            flat[k].imshow(np.full((8, 8, 3), 245, dtype="uint8"))
            flat[k].text(0.5, 0.5, "error", ha="center", va="center",
                         transform=flat[k].transAxes, fontsize=8, color="#c00")
        flat[k].set_xlabel(pretty_style(sstem), fontsize=9.5, color="#333")
        _frame(flat[k], FRAME_RESULT, 1.0)

    # --- celdas sobrantes de la ultima fila ---
    for k in range(n_total, len(flat)):
        flat[k].axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cstem)
    path = out_dir / f"familia_{safe}.png"
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Un PNG por contenido (familia) con todos sus estilos.")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR), help="donde estan las carpetas combined_*")
    ap.add_argument("--content-dir", default=str(CONTENT_DIR), help="donde estan las imagenes de contenido originales")
    ap.add_argument("--out-dir", default=str(FAMILIES_DIR), help="carpeta destino de las familias (PNG por contenido)")
    ap.add_argument("--cols", type=int, default=4, help="columnas antes de envolver a otra fila")
    ap.add_argument("--thumb", type=int, default=512, help="lado mayor (px) de cada miniatura")
    ap.add_argument("--dpi", type=int, default=130, help="resolucion de cada PNG")
    ap.add_argument("--cell", type=float, default=2.9, help="tamano de cada celda en pulgadas")
    args = ap.parse_args()

    output_dir, content_dir, out_dir = Path(args.output_dir), Path(args.content_dir), Path(args.out_dir)
    contents_known = known_stems(content_dir)
    styles_known = known_stems(STYLE_DIR)
    content_files = ({p.stem: p for p in content_dir.iterdir() if p.is_file()}
                     if content_dir.is_dir() else {})

    groups, skipped, in_progress = gather(output_dir, contents_known, styles_known)

    total = sum(len(v) for v in groups.values())
    print(f"[info] {len(groups)} contenidos, {total} resultados clasificados")
    for c in sorted(groups, key=str.lower):
        estilos = ", ".join(pretty_style(s) for s, _ in groups[c])
        mark = "" if c in content_files else "  (sin original)"
        print(f"   - {pretty_content(c)} [{len(groups[c])}]{mark}: {estilos}")
    if in_progress:
        print(f"[info] {len(in_progress)} carpetas aun sin imagen (en proceso?): {in_progress}")
    if skipped:
        print(f"[warn] {len(skipped)} carpetas no reconocidas: {skipped}")

    if not groups:
        print("[!] No hay resultados que dibujar.")
        return

    for cstem in sorted(groups, key=str.lower):
        path = build_familia(cstem, groups[cstem], content_files, out_dir,
                             args.thumb, args.dpi, args.cell, args.cols)
        print(f"   [familia] {path.relative_to(HERE)}")
    print(f"[listo] {len(groups)} familias -> {out_dir.relative_to(HERE)}")


if __name__ == "__main__":
    main()
