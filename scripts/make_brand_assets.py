# -*- coding: utf-8 -*-
"""Gera favicon (SVG/ICO/PNG) e og-image condizentes com a landing page CLT em Movimento.

Paleta da pagina (index.html):
  navy gradiente  #15356b -> #0d1b2a
  dourado (gold)  #fbbf24
  ambar           #c2410c
  texto claro     #dbe4f0 / #9fb0c7

Motivo: barras de saldo de emprego em ascensao + linha de tendencia, sobre o navy.
Roda com Pillow (sem dependencias externas).
"""
import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = ROOT  # gravar na raiz, ao lado do index.html

INK = (13, 27, 42)        # #0d1b2a
NAVY = (21, 53, 107)      # #15356b
GOLD = (251, 191, 36)     # #fbbf24
AMBER = (194, 65, 12)     # #c2410c
LIGHT = (219, 228, 240)   # #dbe4f0
MUT = (159, 176, 199)     # #9fb0c7
WHITE = (255, 255, 255)

WIN_FONTS = r"C:\Windows\Fonts"


def font(name, size):
    try:
        return ImageFont.truetype(os.path.join(WIN_FONTS, name), size)
    except Exception:
        return ImageFont.load_default()


def radial_navy(w, h, cx=0.78, cy=-0.08):
    """Fundo radial navy -> ink, igual ao body da pagina."""
    bg = Image.new("RGB", (w, h), INK)
    px = bg.load()
    # ponto focal do gradiente (78% da largura, um pouco acima do topo)
    fx, fy = cx * w, cy * h
    maxr = math.hypot(max(fx, w - fx), max(fy, h - fy)) * 0.62
    for y in range(h):
        for x in range(w):
            d = math.hypot(x - fx, y - fy) / maxr
            t = max(0.0, min(1.0, d))
            r = int(NAVY[0] + (INK[0] - NAVY[0]) * t)
            g = int(NAVY[1] + (INK[1] - NAVY[1]) * t)
            b = int(NAVY[2] + (INK[2] - NAVY[2]) * t)
            px[x, y] = (r, g, b)
    return bg


def rounded(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


# --------------------------------------------------------------------------
# FAVICON  (icone: barras douradas ascendentes + ponto de tendencia)
# --------------------------------------------------------------------------
def make_favicon_png(size):
    S = size * 8  # supersample para anti-aliasing
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # placa navy arredondada
    pad = int(S * 0.04)
    rounded(d, [pad, pad, S - pad, S - pad], radius=int(S * 0.22), fill=INK + (255,))
    # leve brilho navy no topo
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    rounded(gd, [pad, pad, S - pad, int(S * 0.62)], radius=int(S * 0.22), fill=NAVY + (120,))
    glow = glow.filter(ImageFilter.GaussianBlur(S * 0.06))
    img = Image.alpha_composite(img, glow)
    d = ImageDraw.Draw(img)

    # 4 barras ascendentes
    n = 4
    left = int(S * 0.24)
    right = int(S * 0.78)
    base = int(S * 0.76)
    gap = int(S * 0.045)
    bw = (right - left - gap * (n - 1)) // n
    heights = [0.20, 0.34, 0.50, 0.66]
    tops = []
    for i, hf in enumerate(heights):
        x0 = left + i * (bw + gap)
        top = base - int((base - int(S * 0.26)) * hf / 0.66)
        col = GOLD if i == n - 1 else (251, 191, 36, 150)
        fill = GOLD + (255,) if i == n - 1 else (251, 191, 36, 165)
        rounded(d, [x0, top, x0 + bw, base], radius=int(bw * 0.28), fill=fill)
        tops.append((x0 + bw // 2, top))

    # linha de tendencia ambar conectando topos
    d.line(tops, fill=AMBER + (255,), width=int(S * 0.035), joint="curve")
    # ponto de destaque no ultimo topo
    cx, cy = tops[-1]
    rr = int(S * 0.045)
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=WHITE + (255,))
    d.ellipse([cx - rr + 3, cy - rr + 3, cx + rr - 3, cy + rr - 3], fill=AMBER + (255,))

    return img.resize((size, size), Image.LANCZOS)


def make_favicon_svg():
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <defs>
    <radialGradient id="bg" cx="72%" cy="6%" r="90%">
      <stop offset="0%" stop-color="#15356b"/>
      <stop offset="60%" stop-color="#0d1b2a"/>
    </radialGradient>
  </defs>
  <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#bg)"/>
  <g>
    <rect x="15" y="40" width="7" height="12" rx="2" fill="#fbbf24" opacity="0.65"/>
    <rect x="25" y="33" width="7" height="19" rx="2" fill="#fbbf24" opacity="0.65"/>
    <rect x="35" y="26" width="7" height="26" rx="2" fill="#fbbf24" opacity="0.65"/>
    <rect x="45" y="18" width="7" height="34" rx="2" fill="#fbbf24"/>
  </g>
  <polyline points="18.5,40 28.5,33 38.5,26 48.5,18" fill="none"
            stroke="#c2410c" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="48.5" cy="18" r="3.2" fill="#ffffff"/>
  <circle cx="48.5" cy="18" r="1.7" fill="#c2410c"/>
</svg>
"""


# --------------------------------------------------------------------------
# OG-IMAGE  (1200 x 630)
# --------------------------------------------------------------------------
def make_og():
    W, H = 1200, 630
    img = radial_navy(W, H).convert("RGBA")
    d = ImageDraw.Draw(img)

    # ---- grafico de barras decorativo (lado direito, sutil) ----
    chart = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cd = ImageDraw.Draw(chart)
    bars = [0.30, 0.22, 0.45, 0.38, 0.58, 0.50, 0.72, 0.64, 0.86]
    bx = 690
    bw = 46
    gap = 10
    base = 560
    tops = []
    for i, hf in enumerate(bars):
        x0 = bx + i * (bw + gap)
        top = base - int(360 * hf)
        last = i >= len(bars) - 2
        alpha = 235 if last else 70
        cd.rounded_rectangle([x0, top, x0 + bw, base], radius=8, fill=GOLD + (alpha,))
        tops.append((x0 + bw // 2, top))
    cd.line(tops, fill=AMBER + (220,), width=5, joint="curve")
    cx, cy = tops[-1]
    cd.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=WHITE + (255,))
    cd.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=AMBER + (255,))
    img = Image.alpha_composite(img, chart)
    d = ImageDraw.Draw(img)

    M = 80  # margem esquerda

    # ---- kicker ----
    f_kick = font("segoeuib.ttf", 21)
    kicker = "MERCADO DE TRABALHO FORMAL  ·  NOVO CAGED / MTE"
    # tracking manual (letter-spacing)
    x = M
    ky = 92
    for ch in kicker:
        d.text((x, ky), ch, font=f_kick, fill=GOLD)
        w = d.textlength(ch, font=f_kick)
        x += w + 3

    # ---- titulo serifado ----
    f_h1 = font("georgiab.ttf", 96)
    d.text((M, 140), "CLT em Movimento", font=f_h1, fill=WHITE)

    # ---- thesis ----
    f_th = font("segoeui.ttf", 27)
    thesis = [
        "Seis anos de emprego com carteira assinada no Brasil, do colapso",
        "da pandemia a desaceleracao de 2025, e o que as projecoes e o",
        "debate do fim da escala 6x1 sinalizam para o trabalho formal.",
    ]
    ty = 270
    for ln in thesis:
        d.text((M, ty), ln, font=f_th, fill=LIGHT)
        ty += 40

    # ---- strip de metricas ----
    f_v = font("segoeuib.ttf", 40)
    f_l = font("segoeui.ttf", 17)
    stats = [
        ("10,0 mi", "vinculos liquidos\ndesde jan/2020"),
        ("1,21 mi", "saldo acumulado\nem 12 meses"),
        ("81%", "admissoes a 44h\n(universo da 6x1)"),
    ]
    sx = M
    sy = 430
    col_w = 200
    # linha superior
    d.line([(M, sy - 18), (M + col_w * 3 - 30, sy - 18)], fill=(255, 255, 255, 60), width=1)
    for i, (v, l) in enumerate(stats):
        cxx = sx + i * col_w
        d.text((cxx, sy), v, font=f_v, fill=GOLD if i == 2 else WHITE)
        ly = sy + 52
        for ln in l.split("\n"):
            d.text((cxx, ly), ln, font=f_l, fill=MUT)
            ly += 23
        if i < 2:
            d.line([(cxx + col_w - 24, sy), (cxx + col_w - 24, sy + 92)],
                   fill=(255, 255, 255, 45), width=1)

    # ---- rodape / url ----
    f_ft = font("segoeui.ttf", 19)
    d.text((M, 575), "github.com/avnergomes/clt-brasil", font=f_ft, fill=MUT)
    # previsao tag a direita
    f_tag = font("segoeuib.ttf", 19)
    tag = "previsao -> jun/2027  ·  IC 95% por UF"
    tw = d.textlength(tag, font=f_tag)
    d.text((W - 80 - tw, 575), tag, font=f_tag, fill=GOLD)

    return img.convert("RGB")


def main():
    # SVG
    with open(os.path.join(ASSETS, "favicon.svg"), "w", encoding="utf-8") as fh:
        fh.write(make_favicon_svg())
    print("ok favicon.svg")

    # PNGs + ICO
    sizes = [16, 32, 48, 64, 180, 192, 512]
    pngs = {s: make_favicon_png(s) for s in sizes}
    pngs[32].save(os.path.join(ASSETS, "favicon-32.png"))
    pngs[192].save(os.path.join(ASSETS, "favicon-192.png"))
    pngs[512].save(os.path.join(ASSETS, "favicon-512.png"))
    pngs[180].save(os.path.join(ASSETS, "apple-touch-icon.png"))
    print("ok favicon PNGs")

    ico = pngs[256] if 256 in pngs else make_favicon_png(256)
    ico.save(os.path.join(ASSETS, "favicon.ico"),
             sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    print("ok favicon.ico")

    og = make_og()
    og.save(os.path.join(ASSETS, "og-image.png"), optimize=True)
    print("ok og-image.png", og.size)


if __name__ == "__main__":
    main()
