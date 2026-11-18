"""Render a resume HTML to 300-DPI page images and extract word/region/field
ground truth straight from the browser layout (so boxes match pixels exactly)."""
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright

TEMPLATE_DIR = Path(__file__).parent / "templates"
DPI = 300
SCALE = DPI / 96.0  # CSS px -> image px

PAGINATE_JS = r"""
(maxPages) => {
  const blks = f => [...f.children].filter(c => c.classList.contains('blk'));
  const over = f => f.scrollHeight > f.clientHeight + 1;
  const pagesNow = () => [...document.querySelectorAll('.page')];
  const p1 = pagesNow()[0];
  function makePage(){
    const c = p1.cloneNode(true);
    const n = pagesNow().length + 1;
    c.dataset.page = String(n);
    c.querySelectorAll('.p1only').forEach(e => e.remove());
    c.querySelectorAll('.flow').forEach(f => f.innerHTML = '');
    c.classList.add('cont');
    pagesNow().slice(-1)[0].after(c);
    return c;
  }
  for (let pi = 0; pi < maxPages; pi++) {
    const page = pagesNow()[pi];
    if (!page) break;
    for (const flow of page.querySelectorAll('.flow')) {
      if (!over(flow)) continue;
      let next = pagesNow()[pi + 1];
      if (!next) {
        if (pi + 1 >= maxPages) return {ok:false, reason:'too_long', pages: pagesNow().length};
        next = makePage();
      }
      const nflow = next.querySelector(`.flow[data-flow="${flow.dataset.flow}"]`);
      if (!nflow) return {ok:false, reason:'no_flow_on_next_page'};
      let guard = 0;
      while (over(flow) && guard++ < 500) {
        const b = blks(flow);
        if (b.length <= 1) return {ok:false, reason:'block_too_tall'};
        nflow.insertBefore(b[b.length - 1], nflow.firstChild);
        let b2 = blks(flow);
        while (b2.length > 1 && b2[b2.length - 1].classList.contains('keep')) {
          nflow.insertBefore(b2[b2.length - 1], nflow.firstChild);
          b2 = blks(flow);
        }
      }
    }
  }
  // drop a trailing page that ended up empty
  const last = pagesNow().slice(-1)[0];
  if (pagesNow().length > 1 && ![...last.querySelectorAll('.flow')].some(f => f.children.length)) last.remove();
  // a page whose flows are all empty is not allowed; also a near-empty continuation (orphan heading) is not great but allowed
  return {ok:true, pages: pagesNow().length};
}
"""

EXTRACT_JS = r"""
() => {
  const res = [];
  const pages = [...document.querySelectorAll('.page')];
  for (const [pi, pg] of pages.entries()) {
    const pr = pg.getBoundingClientRect();
    const rel = q => ({x0:q.left - pr.left, y0:q.top - pr.top, x1:q.right - pr.left, y1:q.bottom - pr.top});
    const toks = [], issues = [];
    const walker = document.createTreeWalker(pg, NodeFilter.SHOW_TEXT);
    let node, order = 0;
    while ((node = walker.nextNode())) {
      const txt = node.nodeValue;
      if (!txt || !txt.trim()) continue;
      const el = node.parentElement;
      if (!el || el.closest('style,script,svg')) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === 'hidden' || cs.display === 'none') continue;
      const fe = el.closest('[data-field]');
      const field = fe ? fe.dataset.field : null;
      const blk = el.closest('.blk');
      const re = /\S+/g; let mm;
      while ((mm = re.exec(txt))) {
        const r = document.createRange();
        r.setStart(node, mm.index); r.setEnd(node, mm.index + mm[0].length);
        const rects = [...r.getClientRects()].filter(q => q.width > 0.3 && q.height > 0.3);
        if (!rects.length) continue;
        // a word broken across lines (after a hyphen or a long URL) becomes one token per line fragment
        let pieces = [{a: mm.index, b: mm.index + mm[0].length, rects}];
        if (new Set(rects.map(q => Math.round(q.top))).size > 1) {
          pieces = [];
          let cur = null;
          for (let ci = mm.index; ci < mm.index + mm[0].length; ci++) {
            const cr = document.createRange(); cr.setStart(node, ci); cr.setEnd(node, ci + 1);
            const q = [...cr.getClientRects()].filter(z => z.width > 0.1)[0];
            if (!q) { if (cur) cur.b = ci + 1; continue; }
            if (cur && Math.abs(q.top - cur.top) < 2) { cur.b = ci + 1; cur.rects.push(q); }
            else { cur = {a: ci, b: ci + 1, top: q.top, rects: [q]}; pieces.push(cur); }
          }
        }
        for (const pc of pieces) {
        const prs = pc.rects, word = txt.slice(pc.a, pc.b);
        if (!word.trim()) continue;
        const x0 = Math.min(...prs.map(q => q.left)), y0 = Math.min(...prs.map(q => q.top));
        const x1 = Math.max(...prs.map(q => q.right)), y1 = Math.max(...prs.map(q => q.bottom));
        const box = {x0:x0 - pr.left, y0:y0 - pr.top, x1:x1 - pr.left, y1:y1 - pr.top};
        if (box.x0 < -0.5 || box.y0 < -0.5 || box.x1 > pr.width + 0.5 || box.y1 > pr.height + 0.5) issues.push({type:'outside_page', text: word});
        const cont = el.closest('.flow, td, th, [data-region], .clip');
        if (cont) {
          const cr = cont.getBoundingClientRect();
          if (x1 > cr.right + 1.5 || x0 < cr.left - 1.5) issues.push({type:'h_overflow', text: word, container: cont.className || cont.tagName});
          if (cont.classList.contains('flow') && y1 > cr.bottom + 1) issues.push({type:'v_clip', text: word});
        }
        toks.push({t: word, ...box, field, fs: parseFloat(cs.fontSize), bold: parseInt(cs.fontWeight) >= 600, n: order++,
                   blk: blk ? (blk.dataset.bid || (blk.dataset.bid = 'b' + Math.random().toString(36).slice(2, 9))) : null});
        }
      }
    }
    // regions
    const regions = [];
    pg.querySelectorAll('[data-region]').forEach(el => {
      let q = el.getBoundingClientRect();
      if (el.classList.contains('flow')) {
        const kids = [...el.querySelectorAll('.blk')].map(k => k.getBoundingClientRect()).filter(k => k.height > 0);
        if (!kids.length) return;
        q = {left: Math.min(...kids.map(k => k.left)), top: Math.min(...kids.map(k => k.top)), right: Math.max(...kids.map(k => k.right)), bottom: Math.max(...kids.map(k => k.bottom))};
      }
      if ((q.right - q.left) < 2 || (q.bottom - q.top) < 2) return;
      regions.push({cls: el.dataset.region, ...rel(q)});
    });
    const groups = {};
    pg.querySelectorAll('[data-group]').forEach(el => {
      const q = el.getBoundingClientRect();
      if (q.height < 1 || q.width < 1) return;
      for (const g of el.dataset.group.split(/\s+/)) {
        const u = groups[g] || (groups[g] = {left: 1e9, top: 1e9, right: -1e9, bottom: -1e9});
        u.left = Math.min(u.left, q.left); u.top = Math.min(u.top, q.top); u.right = Math.max(u.right, q.right); u.bottom = Math.max(u.bottom, q.bottom);
      }
    });
    for (const [g, q] of Object.entries(groups)) regions.push({cls: g, ...rel(q)});
    // field texts as displayed
    const fields = [];
    pg.querySelectorAll('[data-field]').forEach(el => {
      fields.push({path: el.dataset.field, text: el.textContent.replace(/\s+/g, ' ').trim()});
    });
    res.push({page: pi + 1, width: pr.width, height: pr.height, tokens: toks, regions, fields, issues});
  }
  return res;
}
"""


class Renderer:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True, trim_blocks=True, lstrip_blocks=True)
        self._pw = sync_playwright().start()
        exe = os.environ.get("CHROMIUM_PATH")
        kw = {"executable_path": exe} if exe else {}
        self.browser = self._pw.chromium.launch(**kw)
        self.ctx = self.browser.new_context(viewport={"width": 900, "height": 2400}, device_scale_factor=SCALE)
        self.page = self.ctx.new_page()

    def html(self, template, person, style):
        return self.env.get_template(f"{template}.html.j2").render(p=person, s=style)

    def render(self, html, max_pages=2):
        pg = self.page
        pg.set_content(html, wait_until="load")
        pg.evaluate("document.fonts.ready.then(() => true)")
        pag = pg.evaluate(PAGINATE_JS, max_pages)
        if not pag.get("ok"):
            return dict(ok=False, reason=pag.get("reason"))
        data = pg.evaluate(EXTRACT_JS)
        issues = [i for d in data for i in d["issues"]]
        if issues:
            return dict(ok=False, reason="layout_issues", issues=issues[:10], pages=data)
        return dict(ok=True, pages=data)

    def shoot(self, paths):
        loc = self.page.locator(".page")
        for i, path in enumerate(paths):
            loc.nth(i).screenshot(path=str(path), animations="disabled", scale="device")

    def close(self):
        try:
            self.ctx.close(); self.browser.close(); self._pw.stop()
        except Exception:
            pass
