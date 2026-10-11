"""Open a generated report in a headless browser, use the controls, and save screenshots.

The report's JavaScript has no unit tests, so run this after changing report/template.html:

    pip install playwright && playwright install chromium     (once, not a project dependency)
    python scripts/check_report_in_browser.py outputs/monthly/forecast_report.html shots/

It prints any JavaScript error and whether the page scrolls sideways at phone width.
Screenshots: desktop (all series), filtered single series with every model line, phone width,
and the Accuracy tab (desktop and phone) after clicking through every metric, period and grouping.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(report: str, out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(Path(report).resolve().as_uri())
        page.screenshot(path=out / "1_total.png", full_page=True)
        for dim in ("market", "product", "company"):
            if page.locator(f"#f_{dim}").count():
                page.select_option(f"#f_{dim}", index=1)
        page.click("[data-mode=series]")
        if page.locator("#msum").is_visible():
            page.click("#msum")
            page.click("text=All models")
            page.click("#msum")
        page.locator("#range").fill("40")
        page.screenshot(path=out / "2_series_models_zoomed.png", full_page=True)
        page.click("[data-view=table]")
        page.set_viewport_size({"width": 390, "height": 800})
        page.click("[data-view=chart]")
        page.screenshot(path=out / "3_phone.png", full_page=True)
        sideways = page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
        # Accuracy tab: every metric, period and grouping once
        page.set_viewport_size({"width": 1100, "height": 900})
        page.click("[data-tab=accuracy]")
        for selector in ("#accMetric button", "#accPeriod button", "#accGroup button", "#accSource button:not([disabled])"):
            for k in range(page.locator(selector).count()):
                page.locator(selector).nth(k).click()
        page.screenshot(path=out / "4_accuracy.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 800})
        page.screenshot(path=out / "5_accuracy_phone.png", full_page=True)
        sideways = sideways or page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth"
        )
        browser.close()
    print("JavaScript errors:", errors or "none")
    print("Sideways scroll at phone width:", sideways)
    return 1 if errors or sideways else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]) if len(sys.argv) >= 3 else print(__doc__) or 2)
