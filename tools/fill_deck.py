"""Fill the result placeholders of the presentation from results/<sim>/ (needs python-pptx).

python tools/fill_deck.py <in.pptx> <out.pptx> [--sim sim_full]
"""
import argparse
import copy
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]


def set_lines(tf, lines, size=None):
    paras = list(tf.paragraphs)
    while len(paras) < len(lines):
        new = copy.deepcopy(paras[-1]._p)
        paras[-1]._p.addnext(new)
        paras = list(tf.paragraphs)
    for p, line in zip(paras, lines):
        runs = p.runs
        if not runs:
            p.add_run()
            runs = p.runs
        runs[0].text = line
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
        f = runs[0].font
        f.italic = False
        if f.color and f.color.type is not None and f.color.rgb == RGBColor(0xC0, 0x00, 0x00):
            f.color.rgb = RGBColor(0x55, 0x55, 0x55)
        if size:
            f.size = Pt(size)
    for p in paras[len(lines):]:
        p._p.getparent().remove(p._p)


def shape(slide, name):
    return next(s for s in slide.shapes if s.name == name)


def replace_with_picture(slide, placeholder, img):
    left, top, w, h = placeholder.left, placeholder.top, placeholder.width, placeholder.height
    iw, ih = Image.open(img).size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(str(img), left + (w - pw) // 2, top + (h - ph) // 2, pw, ph)
    placeholder._element.getparent().remove(placeholder._element)


def add_table_row(table, copy_from):
    tr = copy.deepcopy(table._tbl.tr_lst[copy_from])
    table._tbl.append(tr)
    return table.rows[len(table.rows) - 1]


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--sim", default="sim_full")
    a = ap.parse_args()
    R = ROOT / "results" / a.sim
    F = R / "figures"
    s8 = pd.read_csv(R / "lot5_slide8_table.csv", index_col=0)
    summ = json.loads((R / "lot5_summary.json").read_text())
    met = summ["metrics"]
    cost = pd.read_csv(R / "lot5_cost_table.csv", index_col=[0, 1])
    corner = pd.read_csv(R / "lot5_corner_cases.csv", index_col=[0, 1, 2])
    U, G, D = "U-Net (pre-trained)", "Feature GBM", "Distance to fail"
    prs = Presentation(a.src)
    sl = prs.slides

    s6 = sl[5]
    replace_with_picture(s6, shape(s6, "Rectangle 18"), F / "slide6_riskmap.png")
    pics = [p for p in s6.shapes if p.shape_type == 13]
    slots = {}
    for p in pics:
        slots.setdefault(p.left < 2000000, p)
    for left_slot, img in [(True, "slide6_wm811k.png"), (False, "slide6_inputs.png")]:
        if left_slot in slots and (F / img).exists():
            replace_with_picture(s6, slots[left_slot], F / img)
    for p in pics:
        if p not in slots.values() and p._element.getparent() is not None:
            p._element.getparent().remove(p._element)

    s = sl[7]
    replace_with_picture(s, shape(s, "Rectangle 6"), F / "lot5_recall_vs_burnin.png")
    tbl = shape(s, "Table 7").table
    rows = {"U-Net (proposed)": U, "Distance to fail": D, "Neighbor z-score": "Neighbor z-score",
            "Random": "Random", "No pre-training": "U-Net (no pre-training)"}
    for r in list(tbl.rows)[1:]:
        k = rows[r.cells[0].text.strip()]
        set_lines(r.cells[1].text_frame, [s8.loc[k, "Recall @ 10% burn-in"]])
        set_lines(r.cells[2].text_frame, [s8.loc[k, "Burn-in % @ 90% recall"]])
    r = add_table_row(tbl, 2)
    for c, v in zip(r.cells, ["Feature GBM", s8.loc[G, "Recall @ 10% burn-in"],
                              s8.loc[G, "Burn-in % @ 90% recall"]]):
        set_lines(c.text_frame, [v])
    tshape = shape(s, "Table 7")
    for j, row in enumerate(tbl.rows):
        row.height = Emu(520000 if j == 0 else 380000)
    t_bottom = tshape.top + sum(row.height for row in tbl.rows)
    tb = shape(s, "TextBox 8")
    tb.top = t_bottom + Emu(150000)
    tb.height = Emu(6350000) - tb.top
    u = met[U]
    set_lines(shape(s, "TextBox 8").text_frame, [
        f"Recall ceiling (clued, or within 2 dies of a cluster): {100 * summ['recall_ceiling']:.0f}%; brackets = 95% wafer-bootstrap CI",
        f"Headline: \u201cBurning in 10% of passing dies caught {pct(u['recall@10%burn'])} of latent defects "
        f"(random: 10%); 90% recall needs {pct(u['burn%@90%recall'])} burn-in vs. 90% for random\u201d",
        f"The U-Net ties the hand-engineered Feature GBM ({pct(met[G]['recall@10%burn'])}); "
        f"WM811K pre-training adds no measurable gain",
    ], size=12)

    s = sl[8]
    replace_with_picture(s, shape(s, "Rectangle 10"), F / "slide9_cost_curve.png")
    ce = int(summ["c_escape_slide"])
    cu = cost.loc[(ce, U)]
    hi = cost.loc[(1000, U)]
    set_lines(shape(s, "TextBox 11").text_frame, [
        f"At ${summ['c_burnin']:.2f} burn-in / ${ce:,} escape: burn-in volume reduced by "
        f"{100 - cu['best burn-in %']:.0f}% vs. burn-in-all, {cu['saving vs burn-in-all %']:.0f}% lower cost "
        f"(${cu['burn-in-all cost/wafer'] - cu['cost/wafer at best']:,.0f}/wafer saved)",
        f"Escapes at that threshold: {cu['DPPM at best']:,.0f} DPPM ({pct(cu['recall at best'])} of latent dies caught)",
        f"At \u2265 $1,000 escape cost, burn-in-all is optimal (saving {hi['saving vs burn-in-all %']:.1f}%): a die is worth "
        f"skipping only if its latent probability is below c_burn-in/c_escape ({100 * summ['c_burnin'] / 1000:.2f}%), but even cluster-free wafers "
        f"have {100 * summ['latent_rate_no_cluster']:.2f}% latent in our simulator",
    ], size=13)

    s = sl[9]

    def rec(grouping, group, k=U):
        sub = corner.loc[grouping]
        row = sub[sub.index.get_level_values(0) == group]
        return float(row[k].iloc[0]), int(row.index.get_level_values(1)[0])

    zc, ze = rec("radial zone", "center (r<0.5)"), rec("radial zone", "edge (r>=0.85)")
    zm = rec("radial zone", "middle (0.5-0.85)")
    cl, uc = rec("clue", "clued"), rec("clue", "unclued")
    nr, fa = rec("distance to cluster", "near cluster (<=2)"), rec("distance to cluster", "far / no cluster")
    set_lines(shape(s, "TextBox 7").text_frame, [
        f"Lot 5, U-Net at 10% burn-in: recall by zone center {pct(zc[0])} / middle {pct(zm[0])} / edge {pct(ze[0])}; "
        f"near a cluster {pct(nr[0])} vs. far / no cluster {pct(fa[0])}",
        f"Clued latent dies {pct(cl[0])} caught vs. unclued {pct(uc[0])} ({uc[1]:,} of {cl[1] + uc[1]:,} latent dies "
        f"carry no clue): isolated unclued dies are the dominant escape mode",
    ])

    s = sl[10]
    set_lines(shape(s, "Rounded Rectangle 10").text_frame, [
        "Takeaway",
        f"Burn in {pct(summ['burn_budget'])} of passing dies, catch {pct(u['recall@10%burn'])} of latent defects "
        f"({u['recall@10%burn'] / summ['burn_budget']:.1f}\u00d7 random); selective burn-in saves money only when escapes are cheap relative to burn-in "
        f"(c_escape/c_burn-in \u2272 {int(1 / summ['latent_rate_no_cluster'])})",
    ])
    prs.save(a.dst)
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
