#!/usr/bin/env python3
from pathlib import Path
import os

def generate_res(deca_file: Path, root_folder: Path):
    res_file = deca_file.with_suffix(".res")
    try:
        lines = deca_file.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"[ERR] Erreur lecture {deca_file}: {e}")
        return

    results = []
    in_results = False

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("//"):
            break

        content_clean = stripped[2:].strip()

        if content_clean.lower().startswith("resultats"):
            in_results = True
            continue

        if in_results:
            lower_c = content_clean.lower()
            if lower_c.startswith("historique") or \
                    lower_c.startswith("description") or \
                    lower_c.startswith("test"):
                break
            results.append(content_clean)

    if results:
        while results and not results[-1]:
            results.pop()

        text_content = "\n".join(results) + "\n"
        res_file.write_text(text_content, encoding="utf-8")
        print(f"[OK]  {deca_file}")
    else:
        if res_file.exists():
            res_file.unlink()

cwd = Path.cwd()

base_path = Path("../../..")

target_dirs = [
    base_path / "src/test/deca/codegen/valid",
    base_path / "src/test/deca/codegen/invalid",
    base_path / "src/test/deca/codegen/perf"
]

print(f"--- Génération des fichiers .res ---")
print(f"Exécution depuis : {cwd}\n")

for folder in target_dirs:
    if folder.exists():
        print(f"Scanning: {folder}")
        for deca_file in folder.rglob("*.deca"):
            generate_res(deca_file, folder)
    else:
        alt_folder = Path(str(folder).replace("../../..", "."))
        if alt_folder.exists():
            for deca_file in alt_folder.rglob("*.deca"):
                generate_res(deca_file, alt_folder)
        else:
            print(f"[SKIP] Dossier non trouvé: {folder}")

print(f"\nTerminé !")