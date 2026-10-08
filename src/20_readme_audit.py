from pathlib import Path
import re

ROOT = Path.cwd()
README = ROOT / "README.md"
text = README.read_text(encoding="utf-8-sig")

print("=" * 65)
print("AUDITORIA EDITORIAL - CORRECCION DE ANCLAS GITHUB")
print("=" * 65)

def github_slug(heading):
    heading = re.sub(r"\s+#+\s*$", "", heading).strip()
    heading = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"[^\w\s-]", "", heading.lower())
    heading = re.sub(r"\s", "-", heading)
    return heading

headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.M)

slugs = []
counts = {}
for h in headings:
    base = github_slug(h)
    n = counts.get(base, 0)
    slugs.append(base if n == 0 else f"{base}-{n}")
    counts[base] = n + 1

anchors = re.findall(r"\]\(#([^)]+)\)", text)

errors = []

print("\nENLACES INTERNOS:")
for anchor in anchors:
    ok = anchor in slugs
    print(f"{'OK' if ok else 'ERROR'} | #{anchor}")
    if not ok:
        errors.append(anchor)

images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
missing = [p for p in images if not (ROOT / p).is_file()]

math_count = len(re.findall(r"^\$\$\s*$", text, re.M))

print("\nRESUMEN:")
print("Encabezados:", len(headings))
print("Enlaces internos:", len(anchors))
print("Enlaces incorrectos:", len(errors))
print("Imagenes:", len(images))
print("Imagenes faltantes:", len(missing))
print("Bloques matematicos:", math_count // 2)
print("Nombre interno ausente:", "MOLNIYA" not in text.upper())

assert len(errors) == 0, f"Enlaces incorrectos: {errors}"
assert len(images) == 12
assert not missing
assert math_count % 2 == 0
assert "MOLNIYA" not in text.upper()

print("\nAUDITORIA EDITORIAL: APROBADA")
