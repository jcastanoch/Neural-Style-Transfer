"""Ejecuta NST.py TAL CUAL para varias parejas (contenido, estilo) de forma autonoma.

NO modifica NST.py ni cambia su comportamiento. Por cada pareja:
  1. Crea una copia temporal del fuente con SOLO las dos lineas de nombre de
     imagen cambiadas (CONTENT_IMAGE / STYLE_IMAGE).
  2. Verifica que esa copia difiera del original EXCLUSIVAMENTE en esas dos
     lineas (si hay cualquier otro cambio, aborta por seguridad).
  3. La ejecuta como proceso aparte, con el mismo directorio de trabajo, asi
     que pesos, iteraciones, optimizador, tamano, todo queda identico a NST.py.

Pensado para dejarlo corriendo toda la noche: si una pareja falla por CUALQUIER
motivo (imagen corrupta o ilegible, ruta con caracteres raros, error de CUDA o
de memoria, NST que revienta a mitad, etc.) se registra el traceback completo y
se pasa a la siguiente; NADA tumba el batch. Los logs quedan en batch_logs/.

Uso:
    1) Escribe las parejas en un txt (por defecto pairs.txt), una por linea:
           <contenido> <estilo>
       Ejemplos:
           c1.jpg s1.jpg
           contenidoColombia.jpeg estiloVanGogh.jpg
       - Nombres tal como estan en data/content-images/ y data/style-images/.
       - Separa con espacio (o con coma si algun nombre tiene espacios).
       - Lineas vacias y las que empiezan con # se ignoran.
    2) Corre:
           python batch_nst.py
           python batch_nst.py --pairs mis_parejas.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NST_SOURCE = HERE / "NST.py"
TEMP_SCRIPT = HERE / "_nst_batch_tmp.py"
LOG_DIR = HERE / "batch_logs"
CONTENT_DIR = HERE / "data" / "content-images"
STYLE_DIR = HERE / "data" / "style-images"
OUTPUT_DIR = HERE / "data" / "output-images"

_master_log = None


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if _master_log:
        _master_log.write(line + "\n")
        _master_log.flush()


def parse_pairs(path: Path):
    pairs = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
        parts = [p for p in parts if p]
        if len(parts) != 2:
            log(f"[WARN] linea {lineno} ignorada (se esperaban 2 nombres): {line!r}")
            continue
        pairs.append((parts[0], parts[1]))
    return pairs


def make_modified_source(original: str, content: str, style: str) -> str:
    """Devuelve el fuente de NST.py con SOLO CONTENT_IMAGE y STYLE_IMAGE cambiados."""
    src, n1 = re.subn(r"^CONTENT_IMAGE\s*=.*$", f"CONTENT_IMAGE = {content!r}", original, count=1, flags=re.M)
    src, n2 = re.subn(r"^STYLE_IMAGE\s*=.*$", f"STYLE_IMAGE = {style!r}", src, count=1, flags=re.M)
    if n1 != 1 or n2 != 1:
        raise RuntimeError(f"No ubique CONTENT_IMAGE/STYLE_IMAGE en NST.py (coincidencias: {n1}, {n2}).")

    # Garantia explicita: la unica diferencia con NST.py son esas dos lineas.
    orig_lines, new_lines = original.splitlines(), src.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(orig_lines, new_lines)) if a != b]
    changed_keys = {orig_lines[i].split("=")[0].strip() for i in changed}
    if len(orig_lines) != len(new_lines) or not changed_keys <= {"CONTENT_IMAGE", "STYLE_IMAGE"}:
        raise RuntimeError(f"Cambios inesperados en el fuente (lineas {changed}); aborto por seguridad.")
    return src


def expected_output(content: str, style: str) -> Path:
    # Replica EXACTA del naming de NST.py (basename.split('.')[0]).
    cs = os.path.basename(content).split(".")[0]
    ss = os.path.basename(style).split(".")[0]
    return OUTPUT_DIR / f"combined_{cs}_{ss}" / f"{cs}_{ss}.jpg"


def run_pair(idx: int, total: int, content: str, style: str, original_source: str, timeout: int) -> str:
    log(f"=== [{idx}/{total}] contenido={content}  estilo={style} ===")

    missing = []
    if not (CONTENT_DIR / content).exists():
        missing.append(f"contenido '{content}' no esta en data/content-images/")
    if not (STYLE_DIR / style).exists():
        missing.append(f"estilo '{style}' no esta en data/style-images/")
    if missing:
        for m in missing:
            log(f"[SKIP] {m}")
        return "skip"

    pair_log = LOG_DIR / f"{idx:03d}_{content.split('.')[0]}__{style.split('.')[0]}.log"
    start = datetime.now()
    try:
        # Cualquier paso de aqui (preparar el temp, abrir el log, lanzar NST...)
        # puede reventar; si lo hace lo registramos y devolvemos "fail" SIN
        # tumbar el batch -> el bucle principal sigue con la siguiente pareja.
        TEMP_SCRIPT.write_text(make_modified_source(original_source, content, style), encoding="utf-8")
        with open(pair_log, "w", encoding="utf-8") as out:
            proc = subprocess.run(
                [sys.executable, "-u", str(TEMP_SCRIPT)],   # -u: salida sin buffer (log en vivo)
                cwd=str(HERE), stdout=out, stderr=subprocess.STDOUT,
                timeout=(timeout if timeout > 0 else None),
            )
    except subprocess.TimeoutExpired:
        log(f"[TIMEOUT] supero {timeout}s (log: {pair_log.name})")
        return "timeout"
    except Exception:
        log(f"[FAIL] error preparando/ejecutando la pareja; la ignoro y sigo. Detalle:\n{traceback.format_exc().rstrip()}")
        return "fail"

    dur = str(datetime.now() - start).split(".")[0]
    out_path = expected_output(content, style)
    if proc.returncode == 0 and out_path.exists():
        log(f"[OK] {dur} -> {out_path.relative_to(HERE)}")
        return "ok"
    if proc.returncode == 0:
        log(f"[WARN] exit 0 pero no encontre la salida esperada ({out_path.name}); revisa {pair_log.name}")
        return "warn"
    log(f"[FAIL] exit {proc.returncode} tras {dur}; revisa {pair_log.name}")
    return "fail"


def main() -> None:
    p = argparse.ArgumentParser(description="Corre NST.py para varias parejas (contenido estilo) desde un txt.")
    p.add_argument("--pairs", default="pairs.txt", help="txt con las parejas (una por linea: contenido estilo)")
    p.add_argument("--timeout", type=int, default=0, help="segundos max por pareja (0 = sin limite; util solo para pruebas)")
    args = p.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = HERE / pairs_path
    if not pairs_path.exists():
        print(f"No existe el archivo de parejas: {pairs_path}")
        sys.exit(1)

    LOG_DIR.mkdir(exist_ok=True)
    global _master_log
    _master_log = open(LOG_DIR / "batch.log", "a", encoding="utf-8")

    original_source = NST_SOURCE.read_text(encoding="utf-8")
    pairs = parse_pairs(pairs_path)

    log(f"############### BATCH START - {len(pairs)} parejas ###############")
    if not pairs:
        log("No hay parejas validas en el txt. Nada que hacer.")
        _master_log.close()
        return

    counts = {"ok": 0, "fail": 0, "skip": 0, "warn": 0, "timeout": 0}
    batch_start = datetime.now()
    try:
        for i, (c, s) in enumerate(pairs, 1):
            try:
                result = run_pair(i, len(pairs), c, s, original_source, args.timeout)
            except Exception:  # red de seguridad final: NADA tumba el batch entero
                log(f"[FAIL] excepcion inesperada en ({c}, {s}); la ignoro y sigo. Detalle:\n{traceback.format_exc().rstrip()}")
                result = "fail"
            counts[result] = counts.get(result, 0) + 1
    finally:
        if TEMP_SCRIPT.exists():
            TEMP_SCRIPT.unlink()

    total_dur = str(datetime.now() - batch_start).split(".")[0]
    log(f"############### BATCH DONE - {total_dur} - ok={counts['ok']} fail={counts['fail']} "
        f"skip={counts['skip']} warn={counts['warn']} timeout={counts['timeout']} ###############")
    _master_log.close()


if __name__ == "__main__":
    main()
