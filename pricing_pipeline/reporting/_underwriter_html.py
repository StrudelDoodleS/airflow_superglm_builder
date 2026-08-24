"""Self-contained HTML shell for the underwriter report."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ._underwriter_styles import REPORT_BASE_CSS


@lru_cache(maxsize=1)
def _plotly_js() -> str:
    """Return the installed Plotly runtime escaped for inline embedding."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs().replace("</script", "<\\/script")


def render_underwriter_html(payload: Mapping[str, Any]) -> str:
    """Embed aggregate report data and report-owned CSS in one HTML document."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    title = html.escape(str(payload["metadata"]["title"]), quote=True)
    return (
        _DOCUMENT.replace("__REPORT_TITLE__", title)
        .replace("__REPORT_BASE_CSS__", REPORT_BASE_CSS)
        .replace("__PLOTLY_JS__", _plotly_js())
        .replace("__REPORT_PAYLOAD__", encoded)
    )


_DOCUMENT = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__REPORT_TITLE__</title>
  <style>
__REPORT_BASE_CSS__
  </style>
  <style>
    /* Report-specific composition on top of the report-owned primitives. */
    .report-shell {
      grid-template-areas: "tabs" "context" "view";
      grid-template-rows: auto auto minmax(0, 1fr);
    }
    .report-context {
      grid-area: context;
      min-height: 40px;
      padding: var(--space-1) 0 var(--space-2);
    }
    .report-title {
      max-width: min(440px, 38vw);
      overflow: hidden;
      color: var(--text);
      font-size: 13px;
      font-weight: 600;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    #report-status {
      min-width: 220px;
      margin-left: auto;
      color: var(--muted);
      text-align: right;
    }
    .report-views {
      grid-area: view;
      min-width: 0;
      min-height: 0;
    }
    .review-view {
      display: none;
      min-width: 0;
      min-height: 0;
      height: 100%;
    }
    .review-view.active { display: grid; }
    .report-panel.review-view.active {
      grid-template-rows: auto minmax(0, 1fr);
    }
    .review-workspace-view.active {
      grid-template-rows: auto minmax(0, 1fr);
    }
    .review-toolbar {
      display: flex;
      min-width: 0;
      min-height: 38px;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      padding-bottom: 8px;
    }
    .review-toolbar label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--text);
    }
    .review-toolbar select { min-width: 150px; }
    .review-toolbar .wide-select { min-width: 190px; }
    [hidden] { display: none !important; }
    .review-toolbar-group {
      display: inline-flex;
      min-width: 0;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
    }
    .toggle-button[aria-pressed="true"] {
      border-color: var(--blue);
      background: var(--blue-soft);
      color: var(--blue);
    }
    .review-workspace {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, var(--inspector-width));
      gap: 10px;
      align-items: stretch;
    }
    .review-plot-column {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(360px, 1fr) auto;
      gap: 8px;
    }
    .review-chart-shell {
      min-width: 0;
      min-height: 360px;
      height: 100%;
    }
    .comparison-chart-shell {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .review-chart {
      width: 100%;
      height: 100%;
      min-height: 360px;
    }
    .review-chart > svg,
    .review-chart > .js-plotly-plot {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 360px;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      background: var(--surface);
    }
    .review-inspector {
      height: 100%;
      overflow: hidden;
    }
    .review-inspector .sidepanel-pane { overflow: hidden; }
    .review-inspector .summary-frame { min-height: 0; overflow: auto; }
    #double-lift .sidepanel-pane {
      display: grid;
      min-height: 0;
      flex: 1 1 auto;
      grid-template-rows: auto auto minmax(0, 1fr);
      overflow: hidden;
    }
    .lift-table-scroll {
      min-width: 0;
      min-height: 0;
      max-width: 100%;
      max-height: 100%;
      overflow: auto;
      scrollbar-gutter: stable;
    }
    .lift-table-scroll .report-table { min-width: 760px; }
    .lift-table-scroll .report-table th {
      position: sticky;
      z-index: 1;
      top: 0;
      background: var(--surface-subtle);
    }
    .review-inspector .sidepanel-tabs { padding-right: 4px; }
    .review-note {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
    }
    .report-table-wrap {
      min-width: 0;
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
    }
    .report-table-wrap .report-table th,
    .report-table-wrap .report-table td {
      white-space: nowrap;
    }
    .report-table-wrap .report-table tr:last-child td { border-bottom: 0; }
    .compact-table .report-table {
      width: max-content;
      min-width: 100%;
      table-layout: auto;
      font-size: 11px;
    }
    .compact-table .report-table th,
    .compact-table .report-table td { padding: 5px 6px; }
    #lift-table.lift-table-scroll {
      overflow-x: scroll;
      overflow-y: auto;
      scrollbar-color: #8c959f var(--surface-subtle);
      scrollbar-gutter: stable;
      scrollbar-width: auto;
    }
    #lift-table.lift-table-scroll .report-table { min-width: 760px; }
    #lift-table.lift-table-scroll::-webkit-scrollbar {
      width: 12px;
      height: 12px;
    }
    #lift-table.lift-table-scroll::-webkit-scrollbar-track {
      border: 1px solid var(--border);
      border-radius: 7px;
      background: var(--surface-subtle);
    }
    #lift-table.lift-table-scroll::-webkit-scrollbar-thumb {
      min-width: 36px;
      border: 2px solid var(--surface-subtle);
      border-radius: 7px;
      background: #8c959f;
    }
    #lift-table.lift-table-scroll::-webkit-scrollbar-thumb:hover { background: #57606a; }
    #movement-chart .shapelayer path { pointer-events: none; }
    .importance-bars {
      display: grid;
      gap: 5px;
      padding: 4px 0 8px;
    }
    .importance-row {
      display: grid;
      grid-template-columns: minmax(100px, 150px) minmax(90px, 1fr) 48px;
      align-items: center;
      gap: 7px;
      min-width: 0;
    }
    .importance-name {
      overflow: hidden;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .importance-track {
      height: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 2px;
      background: var(--surface-subtle);
    }
    .importance-fill {
      height: 100%;
      background: var(--blue);
    }
    .importance-value {
      color: var(--muted);
      font-size: 11px;
      text-align: right;
    }
    .report-footer-note {
      padding: 2px 4px 0;
      color: var(--muted);
      font-size: 11px;
    }
    .series-line {
      fill: none;
      stroke-width: 2.3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .series-point {
      fill: var(--surface);
      stroke-width: 1.5;
    }
    .series-point.actual-point { fill: var(--text); stroke: var(--text); }
    .comparison-ci { opacity: 0.13; stroke: none; }
    .density-area { opacity: 0.14; stroke: none; }
    .exposure-label { fill: var(--muted); font-size: 11px; }
    .metric-grid .metric-item {
      grid-template-rows: 28px minmax(20px, auto) auto;
    }
    .metric-grid .metric-item-value { overflow-wrap: anywhere; }
    .metric-grid .metric-item-delta { min-height: 14px; }
    .app-actions .report-action-label {
      max-width: 210px;
      overflow: hidden;
      color: var(--muted);
      font-size: 12px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .interaction-print-shell { display: none; }
    @media (max-width: 1047px) {
      .review-workspace { grid-template-columns: minmax(0, 1fr); }
      .review-inspector { display: none; }
      .report-title { max-width: 52vw; }
    }
    @media (max-height: 620px) {
      .report-shell { height: auto; min-height: 0; }
      .review-workspace-view.active { min-height: 560px; }
    }
    @media print {
      @page { size: A4 landscape; margin: 9mm; }
      html, body {
        width: auto;
        height: auto;
        overflow: visible !important;
      }
      body { padding: 0; background: #fff; }
      .report-shell {
        display: block;
        width: 100%;
        max-width: none;
        height: auto !important;
        min-height: 0;
        overflow: visible !important;
        border: 0;
        box-shadow: none;
      }
      .app-bar,
      .app-actions,
      .report-context,
      .review-toolbar { display: none !important; }
      #interaction-chart { display: none !important; }
      .interaction-screen-shell { display: none !important; }
      .interaction-print-shell { display: block !important; }
      .report-views { display: block; overflow: visible !important; }
      .review-view:not([data-print-page]),
      .review-workspace,
      .review-workspace-view.active .review-workspace {
        display: contents !important;
      }
      [data-print-page] {
        position: relative;
        display: flex !important;
        width: 100%;
        min-height: 190mm !important;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        flex-direction: column;
        break-before: page;
        break-after: page;
        break-inside: avoid-page;
        page-break-before: always;
        page-break-after: always;
        page-break-inside: avoid;
      }
      [data-print-role="summary"],
      [data-print-role="figure"] {
        height: 190mm !important;
        max-height: 190mm !important;
        overflow: hidden !important;
      }
      [data-print-page="overview"] {
        break-before: auto !important;
        page-break-before: auto !important;
      }
      [data-print-page].print-last-page {
        break-after: auto !important;
        page-break-after: auto !important;
      }
      .print-page-header {
        min-height: 28px;
        flex: 0 0 28px;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 8px;
        border-bottom: 1px solid var(--border);
        color: var(--muted);
        font-size: 9px;
      }
      .print-page-header-main {
        display: flex;
        min-width: 0;
        align-items: baseline;
        gap: 10px;
      }
      .print-page-header strong {
        overflow: hidden;
        color: var(--text);
        font-size: 10px;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .print-page-header-section {
        color: var(--text);
        font-size: 13px;
        font-weight: 650;
        white-space: nowrap;
      }
      .print-page-meta { white-space: nowrap; }
      .print-page-footer {
        min-height: 20px;
        flex: 0 0 20px;
        align-items: flex-end;
        justify-content: space-between;
        margin-top: auto;
        padding-top: 6px;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 9px;
      }
      [data-print-role="figure"] { gap: 8px; }
      [data-print-role="figure"] .review-chart-shell,
      [data-print-role="figure"] .review-chart {
        height: 500px !important;
        min-height: 0 !important;
        max-height: 500px !important;
      }
      [data-print-role="figure"] .comparison-chart-shell {
        display: grid !important;
        grid-template-rows: auto minmax(0, 462px);
      }
      [data-print-role="figure"] .comparison-chart-shell .review-chart {
        height: 462px !important;
        max-height: 462px !important;
      }
      .review-chart > svg,
      .review-chart.js-plotly-plot,
      .review-chart .plot-container,
      .review-chart .svg-container,
      .review-chart .main-svg {
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        max-width: 100% !important;
        max-height: 100% !important;
      }
      .modebar-container { display: none !important; }
      .review-inspector,
      .review-inspector .sidepanel-pane,
      .review-inspector .summary-frame,
      #double-lift .sidepanel-pane,
      .lift-table-scroll {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
      }
      .review-inspector {
        display: flex !important;
        position: static !important;
        inset: auto !important;
        width: 100% !important;
        min-width: 0 !important;
        min-height: 190mm !important;
        padding: 0 !important;
        box-shadow: none !important;
        transform: none !important;
        visibility: visible !important;
        transition: none !important;
      }
      .review-inspector .inspector-head { flex: 0 0 auto; }
      .review-inspector .sidepanel-pane,
      #double-lift .sidepanel-pane {
        display: block;
        flex: 0 0 auto;
        padding: 10px 0;
      }
      .compact-table .report-table,
      .lift-table-scroll .report-table {
        width: 100% !important;
        min-width: 0 !important;
        table-layout: auto;
        font-size: 10px;
      }
      .compact-table .report-table th,
      .compact-table .report-table td,
      .lift-table-scroll .report-table th,
      .lift-table-scroll .report-table td {
        padding: 5px 6px !important;
        line-height: 1.25;
        white-space: normal;
      }
      thead { display: table-header-group; }
      tr {
        break-inside: avoid;
        page-break-inside: avoid;
      }
      .lift-table-scroll .report-table th { position: static; }
      [data-print-role="figure"] .metrics-strip { margin-top: 0; }
      .metric-grid { min-height: 0 !important; }
      #double-lift .review-note {
        margin-bottom: 8px;
        font-size: 10px;
        line-height: 1.35;
      }
      #lift-comparison-summary { padding-top: 6px; }
      #lift-comparison-summary .compact-summary { gap: 6px; }
      #lift-comparison-summary .summary-facts {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 5px;
      }
      #lift-comparison-summary .summary-fact {
        padding: 5px 6px;
      }
      #lift-comparison-summary .summary-fact span { font-size: 9px; }
      #lift-comparison-summary .summary-fact strong { font-size: 10px; }
      #lift-comparison-summary .summary-note {
        margin: 0 0 6px;
        font-size: 9px;
        line-height: 1.3;
      }
      #distribution-density-content[hidden],
      #movement-content[hidden],
      #movement-empty[hidden],
      #double-lift-content[hidden],
      #double-lift-empty[hidden] { display: none !important; }
    }
  </style>
</head>
<body>
  <div class="app-shell report-shell">
    <header class="app-bar">
      <div class="app-tabs" role="tablist" aria-label="Model review views">
        <button class="app-tab active" type="button" role="tab" aria-selected="true" data-panel="overview">Overview</button>
        <button class="app-tab" type="button" role="tab" aria-selected="false" data-panel="importance" tabindex="-1">Top features</button>
        <button class="app-tab" type="button" role="tab" aria-selected="false" data-panel="relativities" tabindex="-1">Relativities</button>
        <button class="app-tab" type="button" role="tab" aria-selected="false" data-panel="interactions" tabindex="-1">Interactions</button>
        <button class="app-tab" type="button" role="tab" aria-selected="false" data-panel="distribution" tabindex="-1">Predictions</button>
        <button class="app-tab" type="button" role="tab" aria-selected="false" data-panel="curves" tabindex="-1">Lorenz / gains</button>
        <button class="app-tab" type="button" role="tab" aria-selected="false" data-panel="double-lift" tabindex="-1">Double lift</button>
      </div>
      <div class="app-actions">
        <span class="report-action-label">Read-only aggregate report</span>
        <button id="printAction" class="icon-button" type="button" aria-label="Print report" title="Print report">
          <svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 9V3h12v6"></path><path d="M6 18H4V9h16v9h-2"></path>
            <path d="M6 14h12v7H6z"></path><path d="M17 11h.01"></path>
          </svg>
        </button>
      </div>
    </header>

    <div class="context-bar report-context" role="region" aria-label="Report context">
      <span id="report-title" class="report-title"></span>
      <span id="problem-chip" class="context-chip"></span>
      <span id="power-chip" class="context-chip"></span>
      <span id="rows-chip" class="context-chip"></span>
      <span id="models-chip" class="context-chip"></span>
      <span id="report-status"></span>
    </div>

    <div class="report-views">
      <section class="review-view report-panel active" id="overview" role="tabpanel" data-print-title="Portfolio overview" data-print-page="overview" data-print-role="summary" data-print-section="Portfolio overview">
        <div class="report-header">
          <div>
            <h2>Portfolio overview</h2>
            <p>Weighted diagnostics for the supplied scoring rows. The caller determines whether the sample is genuinely out of time.</p>
          </div>
        </div>
        <div class="report-frame">
          <div class="metrics-strip"><div class="metric-grid" id="overview-metrics"></div></div>
          <section class="report-section">
            <h3>Model metrics</h3>
            <div class="report-note">Lower deviance and higher normalized Gini are better. Observed / predicted near 1 indicates portfolio-level calibration.</div>
            <div class="report-table-wrap" id="metrics-table"></div>
          </section>
          <div class="report-footer-note">This file contains aggregate diagnostics only. Validate source data, holdout design, model purpose and business constraints before taking action.</div>
        </div>
      </section>

      <section class="review-view review-workspace-view" id="importance" role="tabpanel" data-print-title="Top features">
        <div class="review-toolbar">
          <label>Model <select id="importance-model" class="wide-select"></select></label>
          <span class="context-chip">main effects only</span>
          <span id="importance-method-chip" class="context-chip"></span>
        </div>
        <div id="importance-empty" class="review-empty" hidden>Supply model evidence to populate feature importance.</div>
        <div id="importance-content" class="review-workspace">
          <div class="review-plot-column" data-print-page="importance-figure" data-print-role="figure" data-print-section="Top features">
            <div class="chart-shell review-chart-shell"><div class="review-chart" id="importance-chart"></div></div>
            <div class="metrics-strip"><div class="metric-grid" id="importance-metrics"></div></div>
          </div>
          <aside class="inspector review-inspector" aria-label="Top-feature inspector" data-print-page="importance-evidence" data-print-role="evidence" data-print-section="Top-feature evidence">
            <div class="inspector-head"><div class="sidepanel-tabs"><button class="sidepanel-tab active" type="button">Top features</button></div></div>
            <div class="sidepanel-pane">
              <p class="review-note">Ranks features using the method supplied with each model's evidence. It is neither causal nor incremental drop-one importance.</p>
              <div class="summary-frame compact-table" id="importance-table"></div>
            </div>
          </aside>
        </div>
      </section>

      <section class="review-view review-workspace-view" id="relativities" role="tabpanel" data-print-title="Relativities">
        <div class="review-toolbar">
          <label>Term <select id="relativity-feature" class="wide-select"></select></label>
          <span id="relativity-kind" class="context-chip"></span>
          <span id="relativity-edf" class="context-chip"></span>
          <label>Model <select id="relativity-model" class="wide-select"></select></label>
          <button id="relativity-ci" class="toggle-button" type="button" aria-pressed="true">Confidence interval</button>
        </div>
        <div id="relativity-empty" class="review-empty" hidden>Supply normalized main-effect evidence or a fitted-model adapter to show effects.</div>
        <div id="relativity-content" class="review-workspace">
          <div class="review-plot-column" data-print-page="relativities-figure" data-print-role="figure" data-print-section="Relativities">
            <div id="relativity-suppression-note" class="review-empty" hidden></div>
            <div class="chart-shell review-chart-shell"><div class="review-chart" id="relativity-chart"></div></div>
            <div class="metrics-strip"><div class="metric-grid" id="relativity-metrics"></div></div>
          </div>
          <aside class="inspector review-inspector" aria-label="Relativity inspector" data-print-page="relativities-evidence" data-print-role="evidence" data-print-section="Relativity evidence">
            <div class="inspector-head"><div class="sidepanel-tabs"><button class="sidepanel-tab active" type="button">Term</button><button class="sidepanel-tab" type="button" disabled>Evidence</button></div></div>
            <div class="sidepanel-pane">
              <p class="review-note" id="relativity-note">Native fitted relativities with the same baseline, confidence-band and exposure conventions as the SuperGLM editor.</p>
              <div class="summary-frame" id="relativity-inspector"></div>
            </div>
          </aside>
        </div>
      </section>

      <section class="review-view review-workspace-view" id="interactions" role="tabpanel" data-print-title="Interactions">
        <div class="review-toolbar">
          <label>Model <select id="interaction-model" class="wide-select"></select></label>
          <label>Interaction <select id="interaction-term" class="wide-select"></select></label>
          <label>View <select id="interaction-view"></select></label>
          <button id="interaction-ci" class="toggle-button" type="button" aria-pressed="true">Confidence interval</button>
          <details class="model-picker" id="interaction-level-picker" hidden>
            <summary>Levels</summary>
            <div class="model-picker-menu">
              <div class="model-picker-actions">
                <button type="button" data-level-action="defaults">Select defaults</button>
                <button type="button" data-level-action="all">Select all</button>
                <button type="button" data-level-action="none">Deselect all</button>
              </div>
              <div class="model-picker-options" id="interaction-level-options"></div>
            </div>
          </details>
          <span id="interaction-kind" class="context-chip"></span>
        </div>
        <div id="interaction-empty" class="review-empty" hidden>Supply normalized interaction evidence or a fitted model adapter to inspect interactions.</div>
        <div id="interaction-content" class="review-workspace">
          <div class="review-plot-column" data-print-page="interactions-figure" data-print-role="figure" data-print-section="Interactions">
            <div class="chart-shell review-chart-shell interaction-screen-shell"><div class="review-chart" id="interaction-chart"></div></div>
            <div class="chart-shell review-chart-shell interaction-print-shell"><div class="review-chart" id="interaction-print-chart"></div></div>
            <div class="metrics-strip"><div class="metric-grid" id="interaction-metrics"></div></div>
          </div>
          <aside class="inspector review-inspector" aria-label="Interaction inspector" data-print-page="interactions-evidence" data-print-role="evidence" data-print-section="Interaction evidence">
            <div class="inspector-head"><div class="sidepanel-tabs"><button class="sidepanel-tab active" type="button">Evidence</button></div></div>
            <div class="sidepanel-pane">
              <p class="review-note" id="interaction-note"></p>
              <div class="summary-frame" id="interaction-inspector"></div>
            </div>
          </aside>
        </div>
      </section>

      <section class="review-view review-workspace-view" id="distribution" role="tabpanel" data-print-title="Prediction distributions">
        <div class="review-toolbar">
          <div class="mode-segments" id="distribution-view" aria-label="Prediction view">
            <button class="active" type="button" data-view="density">Distribution</button>
            <button type="button" data-view="movement">Model movement</button>
          </div>
          <div class="review-toolbar-group" id="distribution-density-controls">
            <div class="model-picker-wrap">
              <span class="model-picker-label">Models</span>
              <details class="model-picker" id="distribution-model-picker">
                <summary id="distribution-model-summary">All models</summary>
                <div class="model-picker-menu">
                  <div class="model-picker-actions"><button type="button" data-picker-action="all">Select all</button><button type="button" data-picker-action="none">Deselect all</button></div>
                  <div class="model-picker-options" id="distribution-model-options"></div>
                </div>
              </details>
            </div>
            <div class="mode-segments" id="distribution-range" aria-label="Prediction density range">
              <button class="active" type="button" data-range="central">Central 98%</button>
              <button type="button" data-range="full">Full range</button>
            </div>
            <span class="context-chip">weighted distribution</span>
          </div>
          <div class="review-toolbar-group" id="movement-controls" hidden>
            <label>Reference <select id="movement-reference" class="wide-select"></select></label>
            <label>Comparison <select id="movement-comparison" class="wide-select"></select></label>
            <div class="mode-segments" id="movement-view" aria-label="Movement view">
              <button class="active" type="button" data-view="rank">Rank migration</button>
              <button type="button" data-view="level">Prediction levels</button>
            </div>
            <span class="context-chip">aggregate cells only</span>
          </div>
        </div>
        <div class="review-workspace" id="distribution-density-content">
          <div class="review-plot-column" data-print-page="distribution-figure" data-print-role="figure" data-print-section="Prediction distributions">
            <div class="chart-shell review-chart-shell comparison-chart-shell"><div class="chart-legend-strip" id="distribution-legend"></div><div class="review-chart" id="distribution-chart"></div></div>
            <div class="metrics-strip"><div class="metric-grid" id="distribution-metrics"></div></div>
          </div>
          <aside class="inspector review-inspector" aria-label="Prediction inspector" data-print-page="distribution-evidence" data-print-role="evidence" data-print-section="Distribution evidence">
            <div class="inspector-head"><div class="sidepanel-tabs"><button class="sidepanel-tab active" type="button">Distribution</button></div></div>
            <div class="sidepanel-pane">
              <p class="review-note">A weighted Gaussian KDE uses the supplied business weight. Quantiles use that same weight.</p>
              <div class="summary-frame compact-table" id="distribution-inspector"></div>
            </div>
          </aside>
        </div>
        <div id="movement-empty" class="review-empty" hidden>At least two prediction models are required to inspect model movement.</div>
        <div class="review-workspace" id="movement-content" hidden>
          <div class="review-plot-column" data-print-page="movement-figure" data-print-role="figure" data-print-section="Prediction movement">
            <div class="chart-shell review-chart-shell"><div class="review-chart" id="movement-chart"></div></div>
            <div class="metrics-strip"><div class="metric-grid" id="movement-metrics"></div></div>
          </div>
          <aside class="inspector review-inspector" aria-label="Model movement inspector" data-print-page="movement-evidence" data-print-role="evidence" data-print-section="Movement evidence">
            <div class="inspector-head"><div class="sidepanel-tabs"><button class="sidepanel-tab active" type="button">Movement</button></div></div>
            <div class="sidepanel-pane">
              <p class="review-note">Prediction movement is aggregated by business weight. Unsafe cells below the report minimum are omitted before this HTML is written.</p>
              <div class="summary-frame" id="movement-inspector"></div>
            </div>
          </aside>
        </div>
      </section>

      <section class="review-view review-workspace-view" id="curves" role="tabpanel" data-print-title="Lorenz and gains curves">
        <div class="review-toolbar">
          <div class="model-picker-wrap">
            <span class="model-picker-label">Models</span>
            <details class="model-picker" id="curve-model-picker">
              <summary id="curve-model-summary">All models</summary>
              <div class="model-picker-menu">
                <div class="model-picker-actions"><button type="button" data-picker-action="all">Select all</button><button type="button" data-picker-action="none">Deselect all</button></div>
                <div class="model-picker-options" id="curve-model-options"></div>
              </div>
            </details>
          </div>
          <div class="mode-segments" id="curve-mode" aria-label="Curve mode">
            <button class="active" type="button" data-mode="lorenz">Lorenz</button>
            <button type="button" data-mode="gains">Gains</button>
          </div>
          <span id="curve-order-chip" class="context-chip">low to high risk</span>
        </div>
        <div class="review-workspace">
          <div class="review-plot-column" data-print-page="curves-figure" data-print-role="figure" data-print-section="Lorenz and gains">
            <div class="chart-shell review-chart-shell comparison-chart-shell"><div class="chart-legend-strip" id="curve-legend"></div><div class="review-chart" id="curve-chart"></div></div>
            <div class="metrics-strip"><div class="metric-grid" id="curve-metrics"></div></div>
          </div>
          <aside class="inspector review-inspector" aria-label="Lorenz and gains inspector" data-print-page="curves-evidence" data-print-role="evidence" data-print-section="Discrimination evidence">
            <div class="inspector-head"><div class="sidepanel-tabs"><button class="sidepanel-tab active" type="button">Discrimination</button></div></div>
            <div class="sidepanel-pane">
              <p class="review-note">Lorenz and cumulative gains use opposite risk ordering: Lorenz low to high, gains high to low. Curves and Gini are tie-aware; the perfect curve shows the attainable ordering. Its sharp bend is expected for sparse frequency outcomes.</p>
              <div class="summary-frame compact-table" id="curve-table"></div>
            </div>
          </aside>
        </div>
      </section>

      <section class="review-view review-workspace-view" id="double-lift" role="tabpanel" data-print-title="Double lift">
        <div class="review-toolbar">
          <label>Ratio numerator <select id="lift-numerator" class="wide-select"></select></label>
          <label>Ratio denominator <select id="lift-denominator" class="wide-select"></select></label>
          <div class="model-picker-wrap">
            <span class="model-picker-label">Show</span>
            <details class="model-picker" id="lift-model-picker">
              <summary id="lift-model-summary">All models</summary>
              <div class="model-picker-menu">
                <div class="model-picker-actions"><button type="button" data-picker-action="all">Select all</button><button type="button" data-picker-action="none">Deselect all</button></div>
                <div class="model-picker-options" id="lift-model-options"></div>
              </div>
            </details>
          </div>
          <label>Rebase <select id="lift-reference"><option value="none">None</option><option value="actual">Actual</option></select></label>
        </div>
        <div id="double-lift-empty" class="review-empty" hidden>At least two prediction models are required for a double-lift comparison.</div>
        <div id="double-lift-content" class="review-workspace">
          <div class="review-plot-column" data-print-page="double-lift-figure" data-print-role="figure" data-print-section="Double lift">
            <div class="chart-shell review-chart-shell comparison-chart-shell"><div class="chart-legend-strip" id="lift-legend"></div><div class="review-chart" id="lift-chart"></div></div>
            <div class="metrics-strip"><div class="metric-grid" id="lift-metrics"></div></div>
          </div>
          <aside class="inspector review-inspector" aria-label="Double-lift inspector" data-print-page="double-lift-evidence" data-print-role="evidence" data-print-section="Double-lift evidence">
            <div class="inspector-head"><div class="sidepanel-tabs"><button class="sidepanel-tab active" type="button">Bin evidence</button></div></div>
            <div class="sidepanel-pane">
              <p class="review-note" id="lift-cell-note">Bins rank the row-level numerator / denominator. Actual and model rates are calculated as sum(weight × value) / sum(weight); aggregate ratios divide those sums. Rebasing changes display only.</p>
              <div class="summary-frame" id="lift-comparison-summary"></div>
              <div class="summary-frame compact-table lift-table-scroll" id="lift-table" tabindex="0" aria-label="Scrollable double-lift evidence table"></div>
            </div>
          </aside>
        </div>
      </section>
    </div>
  </div>

  <div class="review-tooltip" id="tooltip" role="tooltip"><strong></strong><span></span></div>
  <div class="movement-hover-tooltip" id="movement-hover" role="tooltip" hidden></div>
  <template id="print-page-furniture-template">
    <header class="print-page-header print-page-furniture" aria-hidden="true">
      <div class="print-page-header-main">
        <strong data-print-report-title></strong>
        <span class="print-page-header-section" data-print-header-section></span>
      </div>
      <span class="print-page-meta" data-print-meta></span>
    </header>
    <footer class="print-page-footer print-page-furniture" aria-hidden="true">
      <span data-print-footer-section></span>
      <span data-print-page-number></span>
    </footer>
  </template>
  <script>__PLOTLY_JS__</script>
  <script type="application/json" id="report-data">__REPORT_PAYLOAD__</script>
  <script>
  (() => {
    "use strict";
    const DATA = JSON.parse(document.getElementById("report-data").textContent);
    const NS = "http://www.w3.org/2000/svg";
    const COLORS = ["#0969da", "#d1242f", "#1a7f37", "#8250df", "#bf6a02", "#0550ae", "#cf222e", "#116329"];
    const EXPOSURE_AXIS_HEIGHT = 0.33;
    const MOVEMENT_THERMAL_SCALE = [
      [0.00, "#2b0a3d"],
      [0.20, "#64115f"],
      [0.42, "#b51f58"],
      [0.64, "#ef5a29"],
      [0.82, "#ffb52e"],
      [0.94, "#ffe7a3"],
      [1.00, "#fffdf5"]
    ];
    const INTERACTION_THERMAL_SCALE = [
      [0.00, "#2b0a3d"],
      [0.20, "#64115f"],
      [0.42, "#b51f58"],
      [0.64, "#ef5a29"],
      [0.82, "#ffb52e"],
      [0.94, "#ffe7a3"],
      [1.00, "#fffdf5"]
    ];
    const LIKELIHOOD_LABELS = {
      frequency: "Poisson",
      severity: "Gamma",
      burn_cost: "Tweedie"
    };
    const PLOT_CONFIG = {
      responsive: true,
      displaylogo: false,
      scrollZoom: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "toggleSpikelines"]
    };
    const interactionState = {
      model: null,
      termByModel: new Map(),
      viewByTerm: new Map(),
      levelsByTerm: new Map(),
      ciByTerm: new Map()
    };
    const $ = id => document.getElementById(id);
    const finite = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
    const number = (value, digits = 4) => finite(value) ? Number(value).toLocaleString(undefined, {maximumFractionDigits: digits}) : "—";
    const percent = (value, digits = 1) => finite(value) ? `${(100 * Number(value)).toFixed(digits)}%` : "—";
    const fixedNumber = (value, digits = 4) => finite(value)
      ? Number(value).toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits})
      : "—";
    const fixedPercent = (value, digits = 3) => finite(value) ? `${(100 * Number(value)).toFixed(digits)}%` : "—";
    const color = name => {
      const index = DATA.models.indexOf(name);
      if (index >= 0) return COLORS[index % COLORS.length];
      let hash = 0;
      for (const character of name) hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
      return COLORS[Math.abs(hash) % COLORS.length];
    };
    const element = (tag, attrs = {}, text = null) => {
      const node = document.createElement(tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (text !== null) node.textContent = text;
      return node;
    };
    const svgElement = (tag, attrs = {}, text = null) => {
      const node = document.createElementNS(NS, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (text !== null) node.textContent = text;
      return node;
    };
    const fillSelect = (select, values, {all = false, selected = null} = {}) => {
      select.innerHTML = "";
      if (all) select.appendChild(element("option", {value: "__all__"}, "All models"));
      for (const value of values) select.appendChild(element("option", {value}, value));
      if (selected !== null && (values.includes(selected) || (all && selected === "__all__"))) {
        select.value = selected;
      }
    };
    const modelPicker = (details, options, summary, values, onChange) => {
      const selected = new Set(values);
      const boxes = new Map();
      const update = (notify = true) => {
        const names = values.filter(name => selected.has(name));
        summary.textContent = names.length === values.length
          ? "All models"
          : names.length === 0
            ? "No models"
            : names.length === 1
              ? names[0]
              : `${names.length} of ${values.length} models`;
        for (const [name, box] of boxes) box.checked = selected.has(name);
        if (notify) onChange(names);
      };
      for (const name of values) {
        const label = element("label", {class: "model-picker-option"});
        const box = element("input", {type: "checkbox", value: name, checked: ""});
        boxes.set(name, box);
        box.addEventListener("change", () => {
          if (box.checked) selected.add(name); else selected.delete(name);
          update();
        });
        label.append(box, element("span", {}, name)); options.appendChild(label);
      }
      details.querySelector('[data-picker-action="all"]').addEventListener("click", () => {
        values.forEach(name => selected.add(name)); update();
      });
      details.querySelector('[data-picker-action="none"]').addEventListener("click", () => {
        selected.clear(); update();
      });
      update(false);
      return {values: () => values.filter(name => selected.has(name))};
    };
    const showTooltip = (event, heading, description) => {
      const tip = $("tooltip");
      tip.querySelector("strong").textContent = heading;
      tip.querySelector("span").textContent = description;
      tip.style.display = "block";
      tip.style.left = `${Math.min(event.clientX + 12, window.innerWidth - tip.offsetWidth - 8)}px`;
      tip.style.top = `${Math.min(event.clientY + 12, window.innerHeight - tip.offsetHeight - 8)}px`;
    };
    const hideTooltip = () => { $("tooltip").style.display = "none"; };

    function hideMovementTooltip() {
      $("movement-hover").hidden = true;
    }

    function positionMovementTooltip(pointer) {
      const tip = $("movement-hover");
      const gap = 14;
      const viewportGap = 8;
      let left = Number(pointer.clientX) + gap;
      let top = Number(pointer.clientY) + gap;
      const bounds = tip.getBoundingClientRect();
      if (left + bounds.width > window.innerWidth - viewportGap) {
        left = Number(pointer.clientX) - bounds.width - gap;
      }
      if (top + bounds.height > window.innerHeight - viewportGap) {
        top = Number(pointer.clientY) - bounds.height - gap;
      }
      tip.style.left = `${Math.max(viewportGap, left)}px`;
      tip.style.top = `${Math.max(viewportGap, top)}px`;
    }

    function bindMovementTooltip(container, view, referenceName, comparisonName) {
      const previous = container.__movementTooltipHandlers;
      if (previous) {
        container.removeEventListener("pointermove", previous.pointermove);
        container.removeEventListener("pointerleave", previous.pointerleave);
        if (typeof container.removeListener === "function") {
          container.removeListener("plotly_hover", previous.hover);
          container.removeListener("plotly_unhover", previous.unhover);
        }
      }
      let lastPointer = null;
      const pointermove = event => {
        lastPointer = {clientX: event.clientX, clientY: event.clientY};
        if (!$("movement-hover").hidden) positionMovementTooltip(lastPointer);
      };
      const unhover = () => hideMovementTooltip();
      const hover = event => {
        const point = event.points?.[0];
        const custom = point?.customdata;
        if (!Array.isArray(custom)) {
          hideMovementTooltip();
          return;
        }
        const tip = $("movement-hover");
        const title = view === "rank"
          ? `Reference bin ${point.x} · comparison bin ${point.y}`
          : "Prediction movement";
        const rows = view === "rank"
          ? []
          : [
              ["Reference prediction", fixedNumber(point.x, 4)],
              ["Comparison prediction", fixedNumber(point.y, 4)]
            ];
        rows.push(
          ["Rows", number(custom[0], 0)],
          ["Comparison units", number(custom[1], 0)],
          [DATA.metadata.semantics.volume, fixedNumber(custom[2], 3)],
          [`${DATA.metadata.semantics.volume} share`, fixedPercent(custom[3], 3)],
          [referenceName, fixedNumber(custom[4], 4)],
          [comparisonName, fixedNumber(custom[5], 4)],
          ["Comparison / reference", fixedNumber(custom[6], 4)]
        );
        const grid = element("div", {class: "movement-hover-grid"});
        for (const [label, value] of rows) {
          grid.append(
            element("span", {class: "movement-hover-label"}, label),
            element("span", {class: "movement-hover-value"}, value)
          );
        }
        tip.replaceChildren(element("div", {class: "movement-hover-title"}, title), grid);
        tip.hidden = false;
        const plotBounds = container.getBoundingClientRect();
        const pointer = event.event && finite(event.event.clientX) && finite(event.event.clientY)
          ? event.event
          : lastPointer || {clientX: plotBounds.left + plotBounds.width / 2, clientY: plotBounds.top + plotBounds.height / 2};
        positionMovementTooltip(pointer);
      };
      const pointerleave = () => hideMovementTooltip();
      container.addEventListener("pointermove", pointermove);
      container.addEventListener("pointerleave", pointerleave);
      container.on("plotly_hover", hover);
      container.on("plotly_unhover", unhover);
      container.__movementTooltipHandlers = {pointermove, pointerleave, hover, unhover};
    }

    function renderTable(container, columns, rows) {
      const table = element("table", {class: "report-table"});
      const head = element("thead");
      const header = element("tr");
      for (const column of columns) header.appendChild(element("th", {}, column.label));
      head.appendChild(header); table.appendChild(head);
      const body = element("tbody");
      for (const row of rows) {
        const tr = element("tr");
        for (const column of columns) {
          const value = column.format ? column.format(row[column.key], row) : String(row[column.key] ?? "—");
          tr.appendChild(element("td", {}, value));
        }
        body.appendChild(tr);
      }
      table.appendChild(body); container.replaceChildren(table);
    }

    function renderMetricStrip(container, items) {
      const nodes = [];
      items.forEach((item, index) => {
        if (index) nodes.push(element("div", {class: "metric-divider", "aria-hidden": "true"}));
        const metric = element("div", {class: "metric-item"});
        metric.append(
          element("div", {class: "metric-item-name"}, item.label),
          element("div", {class: "metric-item-value"}, item.value),
          element("div", {class: "metric-item-delta"}, item.detail || "")
        );
        nodes.push(metric);
      });
      container.replaceChildren(...nodes);
    }

    function summaryFacts(container, facts, note = "") {
      const wrapper = element("div", {class: "compact-summary"});
      const grid = element("div", {class: "summary-facts"});
      for (const [label, value] of facts) {
        const fact = element("div", {class: "summary-fact"});
        fact.append(element("span", {}, label), element("strong", {}, value));
        grid.appendChild(fact);
      }
      wrapper.appendChild(grid);
      if (note) wrapper.appendChild(element("p", {class: "summary-note"}, note));
      container.replaceChildren(wrapper);
    }

    function renderLegend(container, series) {
      const entries = series.map((item, index) => {
        const entry = element("div", {class: "chart-legend-entry", title: item.name});
        const swatch = svgElement("svg", {class: "chart-legend-swatch", viewBox: "0 0 24 8", "aria-hidden": "true"});
        swatch.appendChild(svgElement("line", {
          x1: 0, y1: 4, x2: 24, y2: 4,
          stroke: item.color || COLORS[index % COLORS.length],
          "stroke-width": 3,
          "stroke-dasharray": item.dash || ""
        }));
        entry.append(swatch, element("span", {class: "chart-legend-name"}, item.name));
        return entry;
      });
      container.replaceChildren(...entries);
    }

    function renderLegacyXY(container, series, options = {}) {
      container.replaceChildren();
      const validSeries = series.filter(item => item.y && item.y.some(finite));
      if (!validSeries.length) {
        container.appendChild(element("div", {class: "review-empty"}, "No finite chart values are available."));
        return;
      }
      const width = 940, height = 520;
      const margin = {left: 76, right: 76, top: 48, bottom: options.categorical ? 90 : 72};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": options.title || "Chart"});
      const categories = [];
      if (options.categorical) {
        for (const item of validSeries) {
          for (const label of (item.labels || item.x.map(String))) if (!categories.includes(label)) categories.push(label);
        }
      }
      const pointsFor = item => item.y.map((rawY, index) => ({
        x: options.categorical ? categories.indexOf((item.labels || item.x.map(String))[index]) : Number(item.x[index]),
        y: Number(rawY),
        index
      })).filter(point => finite(point.x) && finite(point.y));
      const allPoints = validSeries.flatMap(pointsFor);
      let xMin = options.xMin ?? Math.min(...allPoints.map(point => point.x));
      let xMax = options.xMax ?? Math.max(...allPoints.map(point => point.x));
      if (options.categorical) {
        xMin = options.xMin ?? -0.5;
        xMax = options.xMax ?? Math.max(categories.length - 0.5, 0.5);
      }
      if (xMin === xMax) { xMin -= 0.5; xMax += 0.5; }
      const ciValues = validSeries.flatMap(item => [...(item.ciLower || []), ...(item.ciUpper || [])]).filter(finite).map(Number);
      const yValues = [...allPoints.map(point => point.y), ...ciValues];
      if (finite(options.baseline)) yValues.push(Number(options.baseline));
      if (options.yFromZero) yValues.push(0);
      let yMin = options.yMin ?? Math.min(...yValues);
      let yMax = options.yMax ?? Math.max(...yValues);
      const fixedYDomain = options.yMin !== undefined && options.yMax !== undefined;
      if (yMin === yMax) {
        const pad = Math.max(Math.abs(yMin) * 0.08, 0.05); yMin -= pad; yMax += pad;
      } else if (!fixedYDomain) {
        const pad = (yMax - yMin) * 0.12; yMin -= options.yFromZero ? 0 : pad; yMax += pad;
      }
      const sx = value => margin.left + ((value - xMin) / Math.max(xMax - xMin, 1e-12)) * innerWidth;
      const sy = value => margin.top + innerHeight - ((value - yMin) / Math.max(yMax - yMin, 1e-12)) * innerHeight;
      const pathFor = points => points.map((point, index) => `${index ? "L" : "M"}${sx(point.x).toFixed(2)},${sy(point.y).toFixed(2)}`).join(" ");

      svg.appendChild(svgElement("text", {x: width / 2, y: 24, class: "label", "text-anchor": "middle"}, options.title || ""));
      const exposure = options.exposure;
      if (exposure && exposure.y && exposure.y.some(value => Number(value) > 0)) {
        const weights = exposure.y.map(Number);
        const maxWeight = Math.max(...weights);
        const exposureHeight = innerHeight * EXPOSURE_AXIS_HEIGHT;
        if ((exposure.kind || (options.categorical ? "bars" : "density")) === "bars") {
          const barWidth = Math.max(2, innerWidth / Math.max(categories.length, 1) * 0.82);
          weights.forEach((rawWeight, index) => {
            const label = (exposure.labels || exposure.x.map(String))[index];
            const xValue = categories.indexOf(label);
            if (xValue < 0 || maxWeight <= 0) return;
            const barHeight = exposureHeight * rawWeight / maxWeight;
            svg.appendChild(svgElement("rect", {x: sx(xValue) - barWidth / 2, y: margin.top + innerHeight - barHeight, width: barWidth, height: barHeight, class: "exposure"}));
          });
        } else {
          const exposurePoints = weights.map((rawWeight, index) => ({
            x: Number(exposure.x[index]),
            y: margin.top + innerHeight - exposureHeight * rawWeight / Math.max(maxWeight, 1e-12)
          })).filter(point => finite(point.x));
          if (exposurePoints.length) {
            const d = `M${sx(exposurePoints[0].x)},${margin.top + innerHeight} ` + exposurePoints.map(point => `L${sx(point.x)},${point.y}`).join(" ") + ` L${sx(exposurePoints.at(-1).x)},${margin.top + innerHeight} Z`;
            svg.appendChild(svgElement("path", {d, class: "exposure-density"}));
          }
        }
        const axisX = margin.left + innerWidth;
        svg.appendChild(svgElement("line", {x1: axisX, y1: margin.top + innerHeight - exposureHeight, x2: axisX, y2: margin.top + innerHeight, class: "exposure-axis"}));
        for (const fraction of [0, 0.5, 1]) {
          const y = margin.top + innerHeight - exposureHeight * fraction;
          svg.appendChild(svgElement("line", {x1: axisX, y1: y, x2: axisX + 5, y2: y, class: "exposure-axis"}));
          svg.appendChild(svgElement("text", {x: axisX + 9, y: y + 4, class: "tick-label"}, number(maxWeight * fraction, 3)));
        }
        const label = svgElement("text", {x: width - 18, y: margin.top + innerHeight - exposureHeight / 2, class: "label", "text-anchor": "middle", transform: `rotate(-90 ${width - 18} ${margin.top + innerHeight - exposureHeight / 2})`}, options.exposureLabel || "Exposure");
        svg.appendChild(label);
      }
      for (let index = 0; index <= 5; index++) {
        const value = yMin + (yMax - yMin) * index / 5;
        const y = sy(value);
        svg.appendChild(svgElement("line", {x1: margin.left, y1: y, x2: margin.left + innerWidth, y2: y, class: "grid"}));
        svg.appendChild(svgElement("text", {x: margin.left - 10, y: y + 4, class: "tick-label", "text-anchor": "end"}, number(value, 3)));
      }
      if (finite(options.baseline) && Number(options.baseline) >= yMin && Number(options.baseline) <= yMax) {
        svg.appendChild(svgElement("line", {x1: margin.left, y1: sy(Number(options.baseline)), x2: margin.left + innerWidth, y2: sy(Number(options.baseline)), class: "zero"}));
      }
      svg.appendChild(svgElement("line", {x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + innerHeight, class: "axis"}));
      svg.appendChild(svgElement("line", {x1: margin.left, y1: margin.top + innerHeight, x2: margin.left + innerWidth, y2: margin.top + innerHeight, class: "axis"}));
      const tickCount = options.categorical ? Math.min(categories.length, 14) : 7;
      for (let index = 0; index < tickCount; index++) {
        const position = options.categorical
          ? Math.round(index * (categories.length - 1) / Math.max(tickCount - 1, 1))
          : xMin + (xMax - xMin) * index / Math.max(tickCount - 1, 1);
        const label = options.categorical ? categories[position] : number(position, 3);
        const rotate = options.categorical && (categories.length > 12 || label.length > 12);
        const text = svgElement("text", {x: sx(position), y: margin.top + innerHeight + 20, class: "tick-label x-tick-label", "text-anchor": rotate ? "end" : "middle"}, label.length > 30 ? `${label.slice(0, 27)}…` : label);
        if (rotate) text.setAttribute("transform", `rotate(-38 ${sx(position)} ${margin.top + innerHeight + 20})`);
        svg.appendChild(text);
      }
      svg.appendChild(svgElement("text", {x: width / 2, y: height - 20, class: "label", "text-anchor": "middle"}, options.xLabel || ""));
      svg.appendChild(svgElement("text", {x: 22, y: margin.top + innerHeight / 2, class: "label", "text-anchor": "middle", transform: `rotate(-90 22 ${margin.top + innerHeight / 2})`}, options.yLabel || ""));

      validSeries.forEach((item, seriesIndex) => {
        const points = pointsFor(item);
        const stroke = item.color || COLORS[seriesIndex % COLORS.length];
        if (item.ciLower && item.ciUpper && item.ciLower.length === item.y.length) {
          const lower = points.map(point => ({x: point.x, y: Number(item.ciLower[point.index])})).filter(point => finite(point.y));
          const upper = points.map(point => ({x: point.x, y: Number(item.ciUpper[point.index])})).filter(point => finite(point.y)).reverse();
          if (lower.length === points.length && upper.length === points.length) {
            if (options.categorical) {
              points.forEach(point => {
                const low = Number(item.ciLower[point.index]);
                const high = Number(item.ciUpper[point.index]);
                const x = sx(point.x), lowY = sy(low), highY = sy(high);
                svg.appendChild(svgElement("line", {x1: x, y1: lowY, x2: x, y2: highY, class: "ci-whisker"}));
                svg.appendChild(svgElement("line", {x1: x - 5, y1: lowY, x2: x + 5, y2: lowY, class: "ci-whisker"}));
                svg.appendChild(svgElement("line", {x1: x - 5, y1: highY, x2: x + 5, y2: highY, class: "ci-whisker"}));
              });
            } else {
              const d = `${pathFor(lower)} ${upper.map(point => `L${sx(point.x).toFixed(2)},${sy(point.y).toFixed(2)}`).join(" ")} Z`;
              svg.appendChild(svgElement("path", {d, class: seriesIndex === 0 ? "ci" : "comparison-ci", fill: stroke}));
            }
          }
        }
        if (item.bars) {
          const barWidth = Math.max(2, innerWidth / Math.max(points.length, 1) * 0.82);
          for (const point of points) {
            const top = sy(point.y), bottom = sy(Math.max(0, yMin));
            const barAttributes = {x: sx(point.x) - barWidth / 2, y: Math.min(top, bottom), width: barWidth, height: Math.abs(bottom - top), class: item.exposureBars ? "exposure" : "", opacity: item.exposureBars ? 1 : 0.8};
            if (!item.exposureBars) barAttributes.fill = stroke;
            const bar = svgElement("rect", barAttributes);
            bar.addEventListener("mousemove", event => showTooltip(event, item.name, `${number(point.x, 5)} · ${percent(point.y, 2)}`));
            bar.addEventListener("mouseleave", hideTooltip); svg.appendChild(bar);
          }
        } else {
          if (item.area && points.length) {
            const baseY = sy(Math.max(0, yMin));
            const areaPath = `${pathFor(points)} L${sx(points.at(-1).x).toFixed(2)},${baseY.toFixed(2)} L${sx(points[0].x).toFixed(2)},${baseY.toFixed(2)} Z`;
            svg.appendChild(svgElement("path", {d: areaPath, class: "density-area", fill: stroke}));
          }
          const path = svgElement("path", {d: pathFor(points), class: "series-line", stroke});
          if (item.dash) path.setAttribute("stroke-dasharray", item.dash);
          svg.appendChild(path);
        }
        if (!item.bars && (options.categorical || points.length <= 100) && !item.noPoints) {
          for (const point of points) {
            const circle = svgElement("circle", {cx: sx(point.x), cy: sy(point.y), r: 3.5, class: `series-point${item.actual ? " actual-point" : ""}`, stroke, tabindex: 0});
            const label = options.categorical ? categories[point.x] : number(point.x, 4);
            const detail = `${label} · ${number(point.y, 5)}`;
            circle.addEventListener("mousemove", event => showTooltip(event, item.name, detail));
            circle.addEventListener("mouseleave", hideTooltip);
            circle.addEventListener("focus", event => showTooltip(event, item.name, detail));
            circle.addEventListener("blur", hideTooltip); svg.appendChild(circle);
          }
        }
      });
      if (!options.hideLegend) {
        let legendY = 28;
        const legendX = width - 160;
        validSeries.forEach((item, index) => {
          const stroke = item.color || COLORS[index % COLORS.length];
          svg.appendChild(svgElement("line", {x1: legendX, y1: legendY, x2: legendX + 26, y2: legendY, stroke, "stroke-width": 2.3, "stroke-dasharray": item.dash || ""}));
          svg.appendChild(svgElement("text", {x: legendX + 34, y: legendY + 4, class: "legend"}, item.name));
          legendY += 20;
        });
      }
      container.appendChild(svg);
    }

    function rgba(hex, opacity) {
      const value = String(hex || "#0969da").replace("#", "");
      const resolved = value.length === 3
        ? value.split("").map(character => character + character).join("")
        : value.padEnd(6, "0").slice(0, 6);
      return `rgba(${parseInt(resolved.slice(0, 2), 16)}, ${parseInt(resolved.slice(2, 4), 16)}, ${parseInt(resolved.slice(4, 6), 16)}, ${opacity})`;
    }

    function renderXY(container, series, options = {}) {
      const validSeries = series.filter(item => item.y && item.y.some(finite));
      if (!validSeries.length) {
        if (container.data) Plotly.purge(container);
        container.replaceChildren(element("div", {class: "review-empty"}, "No finite chart values are available."));
        return;
      }
      container.setAttribute("role", "img");
      container.setAttribute("aria-label", options.title || "Chart");
      const traces = [];
      const exposure = options.exposure;
      let maximumExposure = 0;
      if (exposure?.y?.some(value => Number(value) > 0)) {
        maximumExposure = Math.max(...exposure.y.map(Number));
        const exposureKind = exposure.kind || (options.categorical ? "bars" : "density");
        traces.push({
          type: exposureKind === "bars" ? "bar" : "scatter",
          mode: exposureKind === "bars" ? undefined : "lines",
          name: options.exposureLabel || "Exposure",
          x: exposureKind === "bars" ? (exposure.labels || exposure.x.map(String)) : exposure.x,
          y: exposure.y,
          yaxis: "y2",
          marker: exposureKind === "bars" ? {color: "rgba(244, 211, 94, 0.78)", line: {color: "#c99a00", width: 1}} : undefined,
          line: exposureKind === "density" ? {color: "#c99a00", width: 1.2, shape: "spline", smoothing: 0.6} : undefined,
          fill: exposureKind === "density" ? "tozeroy" : undefined,
          fillcolor: exposureKind === "density" ? "rgba(244, 211, 94, 0.68)" : undefined,
          hovertemplate: `${options.exposureLabel || "Exposure"}: %{y:,.3f}<extra></extra>`,
          zorder: 1,
          showlegend: false
        });
      }
      validSeries.forEach((item, seriesIndex) => {
        const stroke = item.color || COLORS[seriesIndex % COLORS.length];
        const x = options.categorical ? (item.labels || item.x.map(String)) : item.x;
        if (!options.categorical && item.ciLower && item.ciUpper && item.ciLower.length === item.y.length) {
          traces.push(
            {
              type: "scatter", mode: "lines", x, y: item.ciLower,
              line: {width: 0, color: stroke}, hoverinfo: "skip", zorder: 2, showlegend: false
            },
            {
              type: "scatter", mode: "lines", x, y: item.ciUpper,
              line: {width: 0, color: stroke}, fill: "tonexty", fillcolor: rgba(stroke, 0.13),
              hoverinfo: "skip", zorder: 2, showlegend: false
            }
          );
        }
        const showMarkers = options.categorical || item.actual || item.y.length <= 100;
        const trace = {
          type: item.bars ? "bar" : "scatter",
          mode: item.bars ? undefined : (showMarkers && !item.noPoints ? "lines+markers" : "lines"),
          name: item.name,
          x,
          y: item.y,
          line: item.bars ? undefined : {color: stroke, width: item.actual ? 2.6 : 2.3, dash: item.dash ? "dash" : "solid"},
          marker: item.bars
            ? {color: stroke}
            : {color: item.actual ? stroke : "#ffffff", line: {color: stroke, width: 1.5}, size: showMarkers ? 6 : 0},
          fill: item.area ? "tozeroy" : undefined,
          fillcolor: item.area ? rgba(stroke, 0.14) : undefined,
          hovertemplate: `${options.categorical ? "%{x}" : "%{x:.4f}"}<br>${item.name}: %{y:.4f}<extra></extra>`,
          connectgaps: false,
          zorder: 3,
          showlegend: !options.hideLegend
        };
        if (options.categorical && item.ciLower && item.ciUpper && item.ciLower.length === item.y.length) {
          trace.error_y = {
            type: "data",
            symmetric: false,
            array: item.ciUpper.map((value, index) => Number(value) - Number(item.y[index])),
            arrayminus: item.ciLower.map((value, index) => Number(item.y[index]) - Number(value)),
            color: rgba(stroke, 0.55),
            thickness: 1.2,
            width: 4,
            visible: true
          };
        }
        traces.push(trace);
      });
      const shapes = [];
      if (finite(options.baseline)) {
        shapes.push({
          type: "line", xref: "paper", x0: 0, x1: 1,
          y0: Number(options.baseline), y1: Number(options.baseline),
          line: {color: "#8c959f", width: 1, dash: "dot"}
        });
      }
      const categories = options.categorical
        ? [...new Set(validSeries.flatMap(item => item.labels || item.x.map(String)))]
        : [];
      const yaxis = {
        title: {text: options.yLabel || "", standoff: 8},
        gridcolor: "#d8dee4",
        zeroline: false,
        rangemode: options.yFromZero ? "tozero" : "normal",
        tickformat: ".4f",
        automargin: true
      };
      if (options.yMin !== undefined && options.yMax !== undefined) {
        yaxis.range = [Number(options.yMin), Number(options.yMax)];
      }
      const layout = {
        autosize: true,
        margin: {l: 76, r: 76, t: 58, b: options.categorical ? 88 : 70},
        title: {text: options.title || "", x: 0.5, xanchor: "center", font: {size: 14}},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size: 11, color: "#24292f"},
        hovermode: "x unified",
        hoverlabel: {bgcolor: "#24292f", font: {color: "#ffffff"}},
        xaxis: {
          title: {text: options.xLabel || "", standoff: 14},
          type: options.categorical ? "category" : "linear",
          categoryorder: options.categorical ? "array" : undefined,
          categoryarray: options.categorical ? categories : undefined,
          tickformat: options.categorical ? undefined : ".4f",
          range: options.categorical
            ? [options.xMin ?? -0.5, options.xMax ?? Math.max(categories.length - 0.5, 0.5)]
            : (options.xMin !== undefined && options.xMax !== undefined ? [Number(options.xMin), Number(options.xMax)] : undefined),
          gridcolor: "#d8dee4",
          zeroline: false,
          automargin: true
        },
        yaxis,
        legend: {
          orientation: "h", x: 1, xanchor: "right", y: 1.12, yanchor: "bottom",
          bgcolor: "rgba(255,255,255,0.72)", font: {size: 11}
        },
        barmode: "overlay",
        bargap: 0.18,
        shapes
      };
      if (maximumExposure > 0) {
        layout.yaxis2 = {
          title: {text: options.exposureLabel || "Exposure", standoff: 8},
          anchor: "x",
          domain: [0, EXPOSURE_AXIS_HEIGHT],
          side: "right",
          range: [0, maximumExposure],
          tickformat: ",.3f",
          nticks: 3,
          showgrid: false,
          zeroline: false,
          automargin: true
        };
      }
      traces.forEach(trace => {
        Object.keys(trace).forEach(key => {
          if (trace[key] === undefined) delete trace[key];
        });
      });
      Plotly.react(container, traces, layout, PLOT_CONFIG)
        .then(() => {
          if (container.closest(".review-view")?.classList.contains("active")) {
            Plotly.Plots.resize(container);
          }
        })
        .catch(error => {
          container.replaceChildren(element("div", {class: "review-empty"}, `Chart rendering failed: ${error.message}`));
        });
    }

    function renderImportanceChart(container, rows, modelName) {
      const width = 940, height = 520;
      const margin = {left: 170, right: 76, top: 48, bottom: 56};
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const maximum = Math.max(...rows.map(row => Number(row.share)), 1e-12);
      const rowHeight = innerHeight / Math.max(rows.length, 1);
      const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Top main effects"});
      svg.appendChild(svgElement("text", {x: width / 2, y: 24, class: "label", "text-anchor": "middle"}, `${modelName} · main-effect importance`));
      for (let index = 0; index <= 5; index++) {
        const value = maximum * index / 5;
        const x = margin.left + innerWidth * index / 5;
        svg.appendChild(svgElement("line", {x1: x, y1: margin.top, x2: x, y2: margin.top + innerHeight, class: "grid"}));
        svg.appendChild(svgElement("text", {x, y: margin.top + innerHeight + 20, class: "tick-label", "text-anchor": "middle"}, percent(value, 3)));
      }
      rows.forEach((row, index) => {
        const y = margin.top + index * rowHeight + rowHeight * 0.2;
        const barHeight = rowHeight * 0.6;
        const barWidth = innerWidth * Number(row.share) / maximum;
        svg.appendChild(svgElement("rect", {x: margin.left, y, width: barWidth, height: barHeight, fill: "var(--blue)", opacity: 0.88}));
        svg.appendChild(svgElement("text", {x: margin.left - 10, y: y + barHeight * 0.68, class: "tick-label", "text-anchor": "end"}, row.feature));
        svg.appendChild(svgElement("text", {x: margin.left + barWidth + 8, y: y + barHeight * 0.68, class: "tick-label"}, percent(row.share, 3)));
      });
      svg.appendChild(svgElement("line", {x1: margin.left, y1: margin.top + innerHeight, x2: margin.left + innerWidth, y2: margin.top + innerHeight, class: "axis"}));
      svg.appendChild(svgElement("text", {x: margin.left + innerWidth / 2, y: height - 14, class: "label", "text-anchor": "middle"}, "Share of total importance"));
      container.replaceChildren(svg);
    }

    function resizePlotsNow(root = document) {
      root.querySelectorAll(".js-plotly-plot").forEach(plot => {
        if (plot.clientWidth > 0 && plot.clientHeight > 0) {
          Plotly.relayout(plot, {width: plot.clientWidth, height: plot.clientHeight});
        }
      });
    }

    function resizePlots(root = document) {
      requestAnimationFrame(() => resizePlotsNow(root));
    }

    function initTabs() {
      const tabs = [...document.querySelectorAll(".app-tab[data-panel]")];
      const activate = (tab, updateHash) => {
        hideMovementTooltip();
        tabs.forEach(item => {
          const selected = item === tab;
          item.classList.toggle("active", selected);
          item.setAttribute("aria-selected", String(selected));
          item.tabIndex = selected ? 0 : -1;
        });
        document.querySelectorAll(".review-view").forEach(panel => panel.classList.toggle("active", panel.id === tab.dataset.panel));
        if (updateHash) history.replaceState(null, "", `#${tab.dataset.panel}`);
        resizePlots(document.getElementById(tab.dataset.panel));
      };
      tabs.forEach(tab => tab.addEventListener("click", () => activate(tab, true)));
      const requested = location.hash.slice(1);
      const initial = tabs.find(tab => tab.dataset.panel === requested);
      if (initial) activate(initial, false);
    }

    function renderOverview() {
      $("report-title").textContent = DATA.metadata.title;
      $("problem-chip").textContent = DATA.metadata.problem_type;
      const problemKey = DATA.metadata.problem_type.toLowerCase().replaceAll(" ", "_");
      const likelihoodLabel = LIKELIHOOD_LABELS[problemKey] || "Tweedie";
      $("power-chip").textContent = `Comparison: ${likelihoodLabel} deviance · p=${number(DATA.metadata.tweedie_power, 3)}`;
      $("rows-chip").textContent = `${Number(DATA.metadata.rows_used).toLocaleString()} rows`;
      $("models-chip").textContent = `${DATA.models.length} model${DATA.models.length === 1 ? "" : "s"}`;
      $("report-status").textContent = `Generated ${DATA.metadata.generated_utc.replace("T", " ")}`;
      const bestDeviance = [...DATA.metrics].sort((left, right) => left.mean_deviance - right.mean_deviance)[0];
      const bestGini = [...DATA.metrics].filter(row => finite(row.normalized_gini)).sort((left, right) => right.normalized_gini - left.normalized_gini)[0];
      renderMetricStrip($("overview-metrics"), [
        {label: "Sample weight", value: number(DATA.metadata.total_weight, 2), detail: `${DATA.metadata.rows_used} positive-weight rows`},
        {label: "Models", value: String(DATA.models.length), detail: DATA.models.join(" · ")},
        {label: "Lowest deviance", value: bestDeviance ? number(bestDeviance.mean_deviance, 5) : "—", detail: bestDeviance?.model || ""},
        {label: "Best normalized Gini", value: bestGini ? number(bestGini.normalized_gini, 4) : "—", detail: bestGini?.model || ""},
        {label: "Zero-weight ignored", value: String(DATA.metadata.zero_weight_rows_ignored), detail: "excluded before aggregation"}
      ]);
      renderTable($("metrics-table"), [
        {key: "model", label: "Model"},
        {key: "mean_deviance", label: "Mean deviance", format: value => number(value, 6)},
        {key: "exact_mean_nll", label: "Exact NLL", format: value => number(value, 6)},
        {key: "pseudo_r2", label: "Pseudo R²", format: value => number(value, 4)},
        {key: "gini", label: "Gini", format: value => number(value, 4)},
        {key: "normalized_gini", label: "Normalized Gini", format: value => number(value, 4)},
        {key: "weighted_actual_mean", label: "Actual", format: value => number(value, 5)},
        {key: "weighted_prediction_mean", label: "Prediction", format: value => number(value, 5)},
        {key: "observed_to_predicted", label: "O / P", format: value => number(value, 4)}
      ], DATA.metrics);
    }

    function renderImportance() {
      const models = Object.keys(DATA.importance).filter(name => DATA.importance[name].length);
      fillSelect($("importance-model"), models);
      const available = models.length > 0;
      $("importance-empty").hidden = available; $("importance-content").hidden = !available;
      if (!available) return;
      const draw = () => {
        const model = $("importance-model").value;
        const rows = DATA.importance[model] || [];
        const method = rows[0]?.method || "";
        $("importance-method-chip").textContent = method || "Method unavailable";
        renderImportanceChart($("importance-chart"), rows, model);
        renderMetricStrip($("importance-metrics"), [
          {label: "Top feature", value: rows[0]?.feature || "—", detail: rows[0] ? percent(rows[0].share) : ""},
          {label: "Features shown", value: String(rows.length), detail: `configured top-k ${DATA.metadata.top_k}`},
          {label: "Top magnitude", value: rows[0] ? number(rows[0].magnitude, 4) : "—", detail: method},
          {label: "Top EDF", value: rows[0] ? number(rows[0].effective_df, 3) : "—", detail: "if available"}
        ]);
        renderTable($("importance-table"), [
          {key: "feature", label: "Feature"},
          {key: "share", label: "Share", format: value => percent(value)},
          {key: "magnitude", label: "Magnitude", format: value => number(value, 4)},
          {key: "effective_df", label: "EDF", format: value => number(value, 3)}
        ], rows);
      };
      $("importance-model").addEventListener("change", draw); draw();
    }

    function renderRelativities() {
      const features = Object.keys(DATA.relativities);
      fillSelect($("relativity-feature"), features);
      const available = features.length > 0;
      $("relativity-empty").hidden = available; $("relativity-content").hidden = !available;
      if (!available) return;
      let selectedModel = "__all__";
      let ciRequested = $("relativity-ci").getAttribute("aria-pressed") === "true";
      const draw = () => {
        const feature = $("relativity-feature").value;
        const all = DATA.relativities[feature] || {};
        selectedModel = $("relativity-model").value;
        const names = selectedModel === "__all__" ? Object.keys(all) : [selectedModel];
        const firstRenderableName = names.find(name => {
          const series = all[name];
          return series
            && series.suppression?.presentation !== "curve_omitted"
            && Array.isArray(series.relativity)
            && series.relativity.length > 0;
        });
        const firstName = firstRenderableName || names.find(name => all[name]);
        const first = all[firstName];
        const suppressionNote = $("relativity-suppression-note");
        const suppressionMessages = {
          partial: "at least one interval did not meet minimum privacy support.",
          all: "no interval met minimum privacy support."
        };
        const suppressedSeries = names.map(name => ({
          name,
          suppression: all[name]?.suppression
        })).filter(item => item.suppression?.presentation === "curve_omitted");
        const suppressionMessage = suppressedSeries.map(item => (
          `Curve omitted for ${item.name} because ${suppressionMessages[item.suppression.status]}`
        )).join(" ");
        suppressionNote.textContent = suppressionMessage || "";
        suppressionNote.hidden = !suppressionMessage;
        const presentation = first?.presentation || {
          title: feature, axis_label: "relativity", reference_value: 1,
          kind_label: first?.kind || "Native fitted component",
          value_label: "Fitted relativity",
          note: "Relativities are native fitted effects; exposure is descriptive context and uses the report sample for fitted objects."
        };
        const ciAvailable = names.length === 1 && Array.isArray(first?.ci_lower) && Array.isArray(first?.ci_upper);
        const showCi = ciRequested && ciAvailable;
        $("relativity-ci").disabled = !ciAvailable;
        $("relativity-ci").setAttribute("aria-disabled", String(!ciAvailable));
        $("relativity-ci").setAttribute("aria-pressed", String(showCi));
        const chartSeries = names.filter(name => all[name]).map((name, index) => ({
          name, x: all[name].x, labels: all[name].labels, y: all[name].relativity,
          ciLower: showCi ? all[name].ci_lower : null,
          ciUpper: showCi ? all[name].ci_upper : null,
          color: index === 0 ? "#0969da" : color(name)
        }));
        const categorical = chartSeries.some(item => Array.isArray(item.labels));
        const exposure = first?.exposure
          ? {...first.exposure, labels: first.labels}
          : first
            ? {kind: categorical ? "bars" : "density", x: first.x, labels: first.labels, y: first.weight}
            : null;
        renderXY($("relativity-chart"), chartSeries, {
          title: presentation.title, xLabel: feature, yLabel: presentation.axis_label,
          baseline: presentation.reference_value, categorical, exposure, exposureLabel: "Exposure"
        });
        $("relativity-kind").textContent = presentation.kind_label;
        $("relativity-edf").textContent = finite(first?.effective_df) ? `EDF ${number(first.effective_df, 3)}` : "EDF —";
        renderMetricStrip($("relativity-metrics"), [
          {label: "Term", value: feature, detail: presentation.kind_label},
          {label: "Model source", value: names.length === 1 ? names[0] : `${names.length} models`, detail: first?.source || ""},
          {label: "Points", value: String(first?.relativity?.length || 0), detail: categorical ? "categorical levels" : "curve grid"},
          {label: "EDF", value: number(first?.effective_df, 3), detail: "effective degrees of freedom"},
          {label: "Confidence interval", value: showCi ? "Shown" : "Hidden", detail: names.length > 1 ? "single-model view only" : ciAvailable ? presentation.value_label : "not available"}
        ]);
        const exposureValues = (first?.exposure?.y || first?.weight || []).filter(finite).map(Number);
        const safeSupport = (first?.density || []).map(row => row.comparison_units).filter(finite).map(Number);
        summaryFacts($("relativity-inspector"), [
          ["Feature", feature], ["Evidence", presentation.kind_label], ["Value", presentation.value_label], ["Model", names.join(", ")],
          ["Source", first?.source || "—"], ["EDF", number(first?.effective_df, 3)],
          ["Exposure total", number(exposureValues.reduce((sum, value) => sum + value, 0), 3)],
          ["Safe support", categorical && safeSupport.length ? `${Math.min(...safeSupport)}–${Math.max(...safeSupport)} distinct comparison units` : "—"],
          ["Suppressed levels", categorical ? String(first?.suppressed_levels || 0) : "—"],
          [
            "Curve suppression",
            suppressedSeries.length
              ? `Omitted for privacy: ${suppressedSeries.map(item => item.name).join(", ")}`
              : "None"
          ]
        ], presentation.note);
        $("relativity-note").textContent = presentation.note;
      };
      const updateModels = () => {
        const models = Object.keys(DATA.relativities[$("relativity-feature").value] || {});
        const nextSelection = selectedModel === "__all__" || models.includes(selectedModel)
          ? selectedModel
          : "__all__";
        fillSelect($("relativity-model"), models, {
          all: models.length > 1,
          selected: models.length === 1 && nextSelection === "__all__" ? models[0] : nextSelection
        });
        selectedModel = $("relativity-model").value;
        draw();
      };
      $("relativity-feature").addEventListener("change", updateModels);
      $("relativity-model").addEventListener("change", event => {
        selectedModel = event.currentTarget.value;
        draw();
      });
      $("relativity-ci").addEventListener("click", () => {
        ciRequested = !ciRequested;
        draw();
      });
      updateModels();
    }

    function interactionPresentation(term) {
      const presentations = {
        native_component: {
          label: "Native fitted component",
          axis: "Relativity",
          note: "A native fitted interaction component on the response scale."
        },
        partial_dependence: {
          label: "Partial dependence",
          axis: "Prediction",
          note: "A response prediction surface, not a fitted rating relativity."
        },
        accumulated_local_effect: {
          label: "Accumulated local effect",
          axis: "Effect",
          note: "A centered accumulated local effect, referenced to zero."
        },
        shap_interaction: {
          label: "SHAP interaction contribution",
          axis: "Effect",
          note: "A signed SHAP interaction contribution, not a fitted rating relativity."
        },
        portfolio_aggregate: {
          label: "Portfolio-conditioned aggregate",
          axis: "Prediction",
          note: "An aggregate over the report portfolio, not a fitted rating relativity."
        }
      };
      return presentations[term?.semantic] || {
        label: "Interaction effect", axis: "Effect", note: "Normalized interaction evidence."
      };
    }

    function interactionViews(term) {
      if (term.plot_kind === "surface") return ["contour_support", "contour", "surface_3d"];
      if (term.plot_kind === "categorical_heatmap") return ["heatmap"];
      if (term.plot_kind === "numeric_categorical") return ["bars"];
      if (term.plot_kind === "numeric_numeric") return ["summary"];
      return ["curves"];
    }

    function interactionMatrix(rows, xValues, yValues, xKey, yKey, valueKey) {
      const indexed = new Map(
        rows.map(row => [JSON.stringify([row[xKey], row[yKey]]), row[valueKey]])
      );
      return yValues.map(y => xValues.map(x => indexed.get(JSON.stringify([x, y])) ?? null));
    }

    function interactionSurfaceTraces(term, view) {
      const x = term.grid_axes.x;
      const y = term.grid_axes.y;
      const z = interactionMatrix(term.effect, x, y, "x", "y", "value");
      const presentation = interactionPresentation(term);
      const hovertemplate = `${term.parents[0]}: %{x:.4f}<br>${term.parents[1]}: %{y:.4f}<br>${presentation.axis}: %{z:.4f}<extra></extra>`;
      if (view === "surface_3d") {
        return [{
          type: "surface", x, y, z,
          colorscale: INTERACTION_THERMAL_SCALE,
          colorbar: {title: {text: presentation.axis}, tickformat: ".4f", thickness: 14},
          hovertemplate,
          contours: {z: {show: true, usecolormap: true, project: {z: true}}},
          lighting: {ambient: 0.72, diffuse: 0.82, roughness: 0.88},
          showscale: true
        }];
      }
      const traces = [{
        type: "contour", x, y, z,
        colorscale: INTERACTION_THERMAL_SCALE,
        colorbar: {title: {text: presentation.axis}, tickformat: ".4f", thickness: 14},
        contours: {coloring: "heatmap", showlabels: false},
        line: {width: 0.45, color: "rgba(36,43,47,0.38)"},
        hovertemplate,
        connectgaps: false,
        zorder: 1
      }];
      if (view === "contour_support" && Array.isArray(term.density)) {
        const hdr = interactionMatrix(term.density, x, y, "x", "y", "hdr_mass");
        [0.50, 0.75, 0.90].forEach((level, index) => traces.push({
          type: "contour", x, y, z: hdr,
          contours: {start: level, end: level, size: 1, coloring: "lines", showlabels: false},
          line: {color: index === 0 ? "#fffdf5" : "#24292f", width: index === 0 ? 2.2 : 1.3, dash: index === 2 ? "dot" : "solid"},
          hoverinfo: "skip",
          showscale: false,
          zorder: 2
        }));
      }
      return traces;
    }

    function interactionHeatmapTraces(term) {
      const left = [...new Set(term.effect.map(row => row.left))];
      const right = [...new Set(term.effect.map(row => row.right))];
      const z = interactionMatrix(term.effect, left, right, "left", "right", "value");
      const supportByCell = new Map(
        (term.support || []).map(row => [JSON.stringify([row.left, row.right]), row])
      );
      const customdata = right.map(rightLevel => left.map(leftLevel => {
        const support = supportByCell.get(JSON.stringify([leftLevel, rightLevel]));
        return support ? [support.comparison_units, support.weight] : [null, null];
      }));
      const presentation = interactionPresentation(term);
      return [{
        type: "heatmap", x: left, y: right, z, customdata,
        colorscale: INTERACTION_THERMAL_SCALE,
        colorbar: {title: {text: presentation.axis}, tickformat: ".4f", thickness: 14},
        hoverongaps: false,
        hovertemplate: `${term.parents[0]}: %{x}<br>${term.parents[1]}: %{y}<br>${presentation.axis}: %{z:.4f}<br>Comparison units: %{customdata[0]:.0f}<br>${DATA.metadata.semantics.volume}: %{customdata[1]:.4f}<extra></extra>`,
        zorder: 2
      }];
    }

    function interactionCurveTraces(term, selectedLevels, showCi) {
      const supportByLevel = new Map((term.support || []).map(row => [String(row.level), row]));
      const traces = [];
      selectedLevels.forEach((level, index) => {
        const rows = term.effect
          .filter(row => String(row.level) === level)
          .sort((left, right) => Number(left.x) - Number(right.x));
        if (!rows.length) return;
        const stroke = COLORS[index % COLORS.length];
        const x = rows.map(row => row.x);
        const y = rows.map(row => row.value);
        if (showCi && rows.every(row => finite(row.lower) && finite(row.upper))) {
          traces.push(
            {
              type: "scatter", mode: "lines", x, y: rows.map(row => row.lower),
              line: {width: 0, color: stroke}, hoverinfo: "skip", showlegend: false, zorder: 2
            },
            {
              type: "scatter", mode: "lines", x, y: rows.map(row => row.upper),
              line: {width: 0, color: stroke}, fill: "tonexty", fillcolor: rgba(stroke, 0.14),
              hoverinfo: "skip", showlegend: false, zorder: 2
            }
          );
        }
        const support = supportByLevel.get(level);
        const customdata = rows.map(() => [support?.comparison_units ?? null, support?.weight ?? null]);
        traces.push({
          type: "scatter", mode: "lines", name: level, x, y, customdata,
          line: {color: stroke, width: 2.4, dash: "solid"},
          hovertemplate: `${term.parents[0]}: %{x:.4f}<br>${term.parents[1]}: ${level}<br>${interactionPresentation(term).axis}: %{y:.4f}<br>Comparison units: %{customdata[0]:.0f}<br>${DATA.metadata.semantics.volume}: %{customdata[1]:.4f}<extra>${level}</extra>`,
          connectgaps: false,
          zorder: 3
        });
      });
      return traces;
    }

    function interactionBarTraces(term) {
      const supportByLevel = new Map((term.support || []).map(row => [String(row.level), row]));
      const rows = term.effect;
      const labels = term.plot_kind === "numeric_numeric"
        ? [term.name]
        : rows.map(row => String(row.level));
      const customdata = rows.map(row => {
        const support = supportByLevel.get(String(row.level));
        return [support?.comparison_units ?? null, support?.weight ?? null];
      });
      const trace = {
        type: "bar", name: interactionPresentation(term).axis,
        x: labels, y: rows.map(row => row.value), customdata,
        marker: {color: "#b51f58", line: {color: "#64115f", width: 1}},
        hovertemplate: `%{x}<br>${interactionPresentation(term).axis}: %{y:.4f}<br>Comparison units: %{customdata[0]:.0f}<br>${DATA.metadata.semantics.volume}: %{customdata[1]:.4f}<extra></extra>`,
        zorder: 3
      };
      if (rows.every(row => finite(row.lower) && finite(row.upper))) {
        trace.error_y = {
          type: "data", symmetric: false,
          array: rows.map(row => Number(row.upper) - Number(row.value)),
          arrayminus: rows.map(row => Number(row.value) - Number(row.lower)),
          color: "#64115f", thickness: 1.2, width: 4, visible: true
        };
      }
      return [trace];
    }

    function interactionLayout(term, view) {
      const presentation = interactionPresentation(term);
      const base = {
        autosize: true,
        margin: {l: 78, r: 96, t: 62, b: 76},
        title: {text: `${term.name} · ${presentation.label}`, x: 0.5, xanchor: "center", font: {size: 14}},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        font: {family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size: 11, color: "#24292f"},
        hovermode: "closest",
        hoverlabel: {bgcolor: "#24292f", bordercolor: "#64115f", font: {color: "#ffffff"}},
        legend: {orientation: "h", x: 1, xanchor: "right", y: 1.10, yanchor: "bottom", bgcolor: "rgba(255,255,255,0.80)"}
      };
      if (view === "surface_3d") {
        base.margin = {l: 20, r: 34, t: 60, b: 20};
        base.scene = {
          xaxis: {title: {text: term.parents[0]}, tickformat: ".4f", gridcolor: "#d8dee4"},
          yaxis: {title: {text: term.parents[1]}, tickformat: ".4f", gridcolor: "#d8dee4"},
          zaxis: {title: {text: presentation.axis}, tickformat: ".4f", gridcolor: "#d8dee4"},
          camera: {eye: {x: 1.45, y: 1.45, z: 1.12}},
          aspectmode: "auto"
        };
        return base;
      }
      base.xaxis = {
        title: {
          text: term.plot_kind === "numeric_numeric"
            ? "Interaction"
            : term.plot_kind === "numeric_categorical"
              ? term.parents[1]
              : term.parents[0],
          standoff: 12
        },
        tickformat: term.plot_kind === "surface" ? ".4f" : undefined,
        gridcolor: "#d8dee4", zeroline: false, automargin: true
      };
      base.yaxis = {
        title: {text: term.plot_kind === "surface" || term.plot_kind === "categorical_heatmap" ? term.parents[1] : presentation.axis, standoff: 10},
        tickformat: ".4f", gridcolor: "#d8dee4", zeroline: false, automargin: true
      };
      if (term.plot_kind === "categorical_heatmap") {
        base.xaxis.title.text = term.parents[0];
        base.yaxis.tickformat = undefined;
      }
      if (term.semantic === "native_component") base.yaxis.rangemode = "normal";
      if (term.semantic === "native_component" && !["surface", "categorical_heatmap"].includes(term.plot_kind)) {
        base.shapes = [{
          type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1,
          line: {color: "#8c959f", width: 1, dash: "dot"}, layer: "below"
        }];
      }
      return base;
    }

    function renderInteractionPlot(container, term, view, selectedLevels, showCi) {
      if (["varying_coefficient", "factor_smooth"].includes(term.plot_kind) && !selectedLevels.length) {
        if (container.data) Plotly.purge(container);
        container.replaceChildren(element("div", {class: "review-empty"}, "Select at least one safe level"));
        return;
      }
      let traces;
      if (term.plot_kind === "surface") traces = interactionSurfaceTraces(term, view);
      else if (term.plot_kind === "categorical_heatmap") traces = interactionHeatmapTraces(term);
      else if (["varying_coefficient", "factor_smooth"].includes(term.plot_kind)) {
        traces = interactionCurveTraces(term, selectedLevels, showCi);
      } else traces = interactionBarTraces(term);
      const layout = interactionLayout(term, view);
      container.setAttribute("role", "img");
      container.setAttribute("aria-label", layout.title.text);
      Plotly.react(container, traces, layout, PLOT_CONFIG)
        .then(() => {
          if (!container.closest("[hidden]") && container.closest(".review-view")?.classList.contains("active")) {
            Plotly.Plots.resize(container);
          }
        })
        .catch(() => {
          container.replaceChildren(element("div", {class: "review-empty"}, "Interaction rendering failed."));
        });
    }

    function renderInteractionPrintChart(term, selectedLevels, showCi) {
      const printView = term.plot_kind === "surface" ? "contour_support" : interactionViews(term)[0];
      renderInteractionPlot($("interaction-print-chart"), term, printView, selectedLevels, showCi);
    }

    function interactionFactValue(value) {
      if (typeof value === "number") return fixedNumber(value, 4);
      if (typeof value === "boolean") return value ? "Yes" : "No";
      return value === null || value === undefined ? "—" : String(value);
    }

    function renderInteractionInspector(container, model, term, selectedLevels) {
      const presentation = interactionPresentation(term);
      const safeCells = Array.isArray(term.support) ? term.support.length : term.effect.length;
      const facts = [
        ["Model", model], ["Interaction", term.name], ["Parents", term.parents.join(" × ")],
        ["Source", term.source], ["Semantic", presentation.label],
        ["Plot kind", term.plot_kind.replaceAll("_", " ")],
        ["Safe cells / levels", String(safeCells)],
        ["Suppressed", "Not retained in report output"]
      ];
      (term.facts || []).forEach(fact => facts.push([fact.label, interactionFactValue(fact.value)]));
      summaryFacts(container, facts, presentation.note);
      const wrapper = container.firstElementChild;
      if (term.warnings?.length) {
        const warnings = element("div", {class: "interaction-warnings"});
        term.warnings.forEach(warning => warnings.appendChild(element("p", {class: "summary-note"}, warning)));
        wrapper.appendChild(warnings);
      }
      if (term.plot_kind === "factor_smooth" && Array.isArray(term.level_diagnostics)) {
        const safe = new Set(term.effect.map(row => String(row.level)));
        const rows = term.level_diagnostics.filter(row => safe.has(String(row.level)));
        const allowed = ["level", "effective_df", "credibility", "has_information", "sufficient_support", "collapsed"];
        const present = allowed.filter(column => rows.some(row => Object.hasOwn(row, column)));
        const tableFrame = element("div", {class: "compact-table interaction-diagnostics"});
        renderTable(tableFrame, present.map(column => ({
          key: column,
          label: column.replaceAll("_", " "),
          format: value => typeof value === "number" ? fixedNumber(value, 4) : interactionFactValue(value)
        })), rows);
        wrapper.appendChild(tableFrame);
      }
      if (selectedLevels.length) {
        wrapper.appendChild(element("p", {class: "summary-note"}, `Showing ${selectedLevels.length} safe level${selectedLevels.length === 1 ? "" : "s"}.`));
      }
    }

    function renderInteractionLevelOptions(term, key, redraw) {
      const picker = $("interaction-level-picker");
      const options = $("interaction-level-options");
      const safeLevels = [...new Set(term.effect.filter(row => Object.hasOwn(row, "level")).map(row => String(row.level)))];
      picker.hidden = safeLevels.length === 0;
      if (!safeLevels.length) {
        options.replaceChildren();
        return [];
      }
      if (!interactionState.levelsByTerm.has(key)) {
        const defaults = (term.default_levels || []).map(String).filter(level => safeLevels.includes(level)).slice(0, 6);
        interactionState.levelsByTerm.set(key, new Set(defaults.length ? defaults : safeLevels.slice(0, 6)));
      }
      const selected = interactionState.levelsByTerm.get(key);
      [...selected].forEach(level => { if (!safeLevels.includes(level)) selected.delete(level); });
      const nodes = safeLevels.map(level => {
        const label = element("label", {class: "model-picker-option"});
        const box = element("input", {type: "checkbox", value: level});
        box.checked = selected.has(level);
        box.addEventListener("change", () => {
          if (box.checked) selected.add(level); else selected.delete(level);
          redraw();
        });
        label.append(box, element("span", {}, level));
        return label;
      });
      options.replaceChildren(...nodes);
      picker.querySelector("summary").textContent = selected.size === safeLevels.length
        ? "All safe levels"
        : selected.size === 0
          ? "No levels"
          : `${selected.size} of ${safeLevels.length} levels`;
      picker.querySelector('[data-level-action="defaults"]').onclick = () => {
        const defaults = (term.default_levels || []).map(String).filter(level => safeLevels.includes(level)).slice(0, 6);
        interactionState.levelsByTerm.set(key, new Set(defaults.length ? defaults : safeLevels.slice(0, 6)));
        redraw();
      };
      picker.querySelector('[data-level-action="all"]').onclick = () => {
        interactionState.levelsByTerm.set(key, new Set(safeLevels)); redraw();
      };
      picker.querySelector('[data-level-action="none"]').onclick = () => {
        interactionState.levelsByTerm.set(key, new Set()); redraw();
      };
      return safeLevels.filter(level => selected.has(level));
    }

    function fillInteractionView(select, views, selected) {
      const labels = {
        contour_support: "Contour + support", contour: "Contour", surface_3d: "3D surface",
        heatmap: "Heatmap", curves: "Curves", bars: "Bars", summary: "Summary"
      };
      select.replaceChildren(...views.map(view => element("option", {value: view}, labels[view] || view)));
      select.value = views.includes(selected) ? selected : views[0];
    }

    function renderInteractions() {
      const models = Object.keys(DATA.interactions?.models || {});
      const available = models.length > 0;
      $("interaction-empty").hidden = available;
      $("interaction-content").hidden = !available;
      if (!available) return;

      const modelSelect = $("interaction-model");
      const termSelect = $("interaction-term");
      const viewSelect = $("interaction-view");
      interactionState.model = models.includes(interactionState.model)
        ? interactionState.model
        : models[0];
      fillSelect(modelSelect, models, {selected: interactionState.model});

      const keyFor = (model, term) => `${model}\u0000${term}`;
      const draw = () => {
        const model = interactionState.model;
        const name = termSelect.value;
        const term = DATA.interactions.models[model]?.[name];
        if (!term) return;
        interactionState.termByModel.set(model, name);
        const key = keyFor(model, name);
        const views = interactionViews(term);
        const rememberedView = interactionState.viewByTerm.get(key);
        const selectedView = views.includes(rememberedView) ? rememberedView : views[0];
        fillInteractionView(viewSelect, views, selectedView);
        interactionState.viewByTerm.set(key, viewSelect.value);
        const hasCi = term.effect.some(row => finite(row.lower) && finite(row.upper));
        const ciRequested = interactionState.ciByTerm.has(key)
          ? interactionState.ciByTerm.get(key)
          : true;
        const showCi = Boolean(ciRequested && hasCi);
        $("interaction-ci").disabled = !hasCi;
        $("interaction-ci").setAttribute("aria-disabled", String(!hasCi));
        $("interaction-ci").setAttribute("aria-pressed", String(showCi));
        const presentation = interactionPresentation(term);
        const selectedLevels = renderInteractionLevelOptions(term, key, draw);
        $("interaction-kind").textContent = presentation.label;
        $("interaction-note").textContent = presentation.note;
        renderInteractionPlot($("interaction-chart"), term, viewSelect.value, selectedLevels, showCi);
        renderInteractionPrintChart(term, selectedLevels, showCi);
        renderMetricStrip($("interaction-metrics"), [
          {label: "Interaction", value: term.name, detail: term.parents.join(" × ")},
          {label: "Evidence", value: presentation.label, detail: term.source},
          {
            label: "Screen view",
            value: viewSelect.value.replaceAll("_", " "),
            detail: viewSelect.value === "surface_3d" ? "print: contour + support" : presentation.axis
          },
          {label: "Cells", value: String(term.effect.length), detail: "privacy-safe normalized cells"}
        ]);
        renderInteractionInspector($("interaction-inspector"), model, term, selectedLevels);
      };

      const updateTerms = preferred => {
        const model = interactionState.model;
        const terms = Object.keys(DATA.interactions.models[model] || {});
        const remembered = interactionState.termByModel.get(model);
        const selected = terms.includes(preferred)
          ? preferred
          : terms.includes(remembered)
            ? remembered
            : terms[0];
        fillSelect(termSelect, terms, {selected});
        draw();
      };

      modelSelect.addEventListener("change", () => {
        const sharedTerm = termSelect.value;
        interactionState.model = modelSelect.value;
        updateTerms(sharedTerm);
      });
      termSelect.addEventListener("change", draw);
      viewSelect.addEventListener("change", () => {
        const key = keyFor(interactionState.model, termSelect.value);
        interactionState.viewByTerm.set(key, viewSelect.value);
        draw();
      });
      $("interaction-ci").addEventListener("click", () => {
        const key = keyFor(interactionState.model, termSelect.value);
        const current = $("interaction-ci").getAttribute("aria-pressed") === "true";
        interactionState.ciByTerm.set(key, !current);
        draw();
      });
      updateTerms(interactionState.termByModel.get(interactionState.model));
    }

    function renderMovementHeatmap(container, block, view, referenceName, comparisonName) {
      const source = block[view];
      if (!source.cells.length) {
        if (container.data) Plotly.purge(container);
        container.replaceChildren(element("div", {class: "review-empty"}, `No ${view === "rank" ? "rank-migration" : "prediction-level"} cells satisfy the minimum of ${DATA.metadata.minimum_cell_size} comparison units.`));
        return;
      }
      const x = view === "rank" ? source.x_labels : source.x_values;
      const y = view === "rank" ? source.y_labels : source.y_values;
      const z = y.map(() => x.map(() => null));
      const customdata = y.map(() => x.map(() => null));
      source.cells.forEach(cell => {
        z[cell.y - 1][cell.x - 1] = Number(cell.weight_share);
        customdata[cell.y - 1][cell.x - 1] = [
          cell.rows, cell.comparison_units, cell.weight, cell.weight_share,
          cell.reference_prediction, cell.comparison_prediction, cell.prediction_ratio
        ];
      });
      const traces = [{
        type: "heatmap",
        x,
        y,
        z,
        customdata,
        colorscale: MOVEMENT_THERMAL_SCALE,
        zmin: 0,
        zmax: Math.max(...source.cells.map(cell => Number(cell.weight_share))),
        colorbar: {title: {text: `${DATA.metadata.semantics.volume}<br>share`}, tickformat: ".3%", thickness: 14},
        hoverongaps: false,
        hoverinfo: "none"
      }];
      let referenceShape;
      if (view === "rank") {
        const diagonal = Math.min(x.length, y.length);
        referenceShape = {
          type: "line", xref: "x", yref: "y", layer: "above",
          x0: x[0], y0: y[0], x1: x[diagonal - 1], y1: y[diagonal - 1],
          line: {color: "#24292f", width: 1.5, dash: "dash"}
        };
      } else {
        const allValues = [...x, ...y].map(Number);
        const minimum = Math.min(...allValues);
        const maximum = Math.max(...allValues);
        referenceShape = {
          type: "line", xref: "x", yref: "y", layer: "above",
          x0: minimum, y0: minimum, x1: maximum, y1: maximum,
          line: {color: "#24292f", width: 1.5, dash: "dash"}
        };
      }
      const layout = {
        autosize: true,
        margin: {l: 82, r: 104, t: 58, b: 76},
        title: {text: view === "rank" ? `${referenceName} → ${comparisonName} rank migration` : `${referenceName} → ${comparisonName} prediction movement`, x: 0.5, xanchor: "center", font: {size: 14}},
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#eaeef2",
        font: {family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size: 11, color: "#24292f"},
        xaxis: {
          title: {text: view === "rank" ? `${referenceName} weighted risk bin` : `${referenceName} prediction`, standoff: 14},
          type: view === "rank" ? "category" : "log",
          categoryorder: view === "rank" ? "array" : undefined,
          categoryarray: view === "rank" ? x : undefined,
          tickformat: view === "rank" ? undefined : ".4f",
          gridcolor: "#d8dee4", zeroline: false, automargin: true
        },
        yaxis: {
          title: {text: view === "rank" ? `${comparisonName} weighted risk bin` : `${comparisonName} prediction`, standoff: 12},
          type: view === "rank" ? "category" : "log",
          categoryorder: view === "rank" ? "array" : undefined,
          categoryarray: view === "rank" ? y : undefined,
          tickformat: view === "rank" ? undefined : ".4f",
          gridcolor: "#d8dee4", zeroline: false, automargin: true
        },
        hovermode: "closest",
        shapes: [referenceShape],
        showlegend: false
      };
      container.setAttribute("role", "img");
      container.setAttribute("aria-label", layout.title.text);
      Plotly.react(container, traces, layout, PLOT_CONFIG)
        .then(() => {
          bindMovementTooltip(container, view, referenceName, comparisonName);
          if (container.closest(".review-view")?.classList.contains("active")) {
            Plotly.Plots.resize(container);
          }
        })
        .catch(error => {
          container.replaceChildren(element("div", {class: "review-empty"}, `Movement rendering failed: ${error.message}`));
        });
    }

    function renderDistribution() {
      let range = "central";
      let selectedNames = [...DATA.models];
      let reportView = "density";
      let movementView = "rank";
      const draw = () => {
        if (!selectedNames.length) {
          $("distribution-chart").replaceChildren(element("div", {class: "review-empty"}, "No models selected"));
          $("distribution-legend").replaceChildren();
          $("distribution-metrics").replaceChildren();
          $("distribution-inspector").replaceChildren(element("div", {class: "review-empty"}, "Select one or more models to compare prediction densities."));
          return;
        }
        const lower = range === "central"
          ? Math.min(...selectedNames.map(name => Number(DATA.distributions[name].quantiles.p01)))
          : -Infinity;
        const upper = range === "central"
          ? Math.max(...selectedNames.map(name => Number(DATA.distributions[name].quantiles.p99)))
          : Infinity;
        const series = selectedNames.map(name => {
          const distribution = DATA.distributions[name];
          let indices = distribution.x.map((_value, index) => index).filter(index => Number(distribution.x[index]) >= lower && Number(distribution.x[index]) <= upper);
          if (indices.length < 2) indices = distribution.x.map((_value, index) => index);
          return {
            name,
            x: indices.map(index => distribution.x[index]),
            y: indices.map(index => distribution.density[index]),
            color: color(name),
            noPoints: true,
            area: true
          };
        });
        renderLegend($("distribution-legend"), series);
        renderXY($("distribution-chart"), series, {
          title: `Prediction density${range === "central" ? " · shared p01–p99 range" : " · full range"}`,
          xLabel: DATA.metadata.semantics.prediction,
          yLabel: "Weighted density",
          xMin: range === "central" ? lower : undefined,
          xMax: range === "central" ? upper : undefined,
          yFromZero: true,
          hideLegend: true
        });
        const medians = selectedNames.map(name => ({name, value: Number(DATA.distributions[name].quantiles.p50)}));
        const lowest = [...medians].sort((left, right) => left.value - right.value)[0];
        const highest = [...medians].sort((left, right) => right.value - left.value)[0];
        renderMetricStrip($("distribution-metrics"), [
          {label: "Models selected", value: String(selectedNames.length), detail: selectedNames.join(" · ")},
          {label: "Range", value: range === "central" ? "Central 98%" : "Full", detail: "one shared axis"},
          {label: "Lowest median", value: number(lowest.value, 5), detail: lowest.name},
          {label: "Highest median", value: number(highest.value, 5), detail: highest.name},
          {label: "Weight basis", value: DATA.metadata.semantics.volume, detail: "business-weighted KDE"}
        ]);
        renderTable($("distribution-inspector"), [
          {key: "model", label: "Model"},
          {key: "p01", label: "P01", format: value => number(value, 5)},
          {key: "p10", label: "P10", format: value => number(value, 5)},
          {key: "p50", label: "P50", format: value => number(value, 5)},
          {key: "p90", label: "P90", format: value => number(value, 5)},
          {key: "p99", label: "P99", format: value => number(value, 5)},
          {key: "bandwidth", label: "Bandwidth", format: value => number(value, 6)},
          {key: "effective_rows", label: "Eff. rows", format: value => number(value, 1)}
        ], selectedNames.map(name => ({
          model: name,
          ...DATA.distributions[name].quantiles,
          bandwidth: DATA.distributions[name].bandwidth,
          effective_rows: DATA.distributions[name].effective_rows
        })));
      };
      modelPicker($("distribution-model-picker"), $("distribution-model-options"), $("distribution-model-summary"), DATA.models, names => {
        selectedNames = names; draw();
      });
      $("distribution-range").querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
        range = button.dataset.range;
        $("distribution-range").querySelectorAll("button").forEach(item => item.classList.toggle("active", item === button));
        draw();
      }));
      const movementAvailable = DATA.models.length >= 2;
      fillSelect($("movement-reference"), DATA.models, {selected: DATA.models[0]});
      fillSelect($("movement-comparison"), DATA.models, {selected: DATA.models.at(-1)});
      $("distribution-view").querySelector('[data-view="movement"]').disabled = !movementAvailable;
      const drawMovement = () => {
        if (!movementAvailable) return;
        const referenceName = $("movement-reference").value;
        const comparisonName = $("movement-comparison").value;
        const block = DATA.movement[referenceName][comparisonName];
        const source = block[movementView];
        renderMovementHeatmap($("movement-chart"), block, movementView, referenceName, comparisonName);
        const summary = block.summary;
        renderMetricStrip($("movement-metrics"), [
          {label: "Log-prediction correlation", value: number(summary.weighted_log_prediction_correlation, 4), detail: "business-weighted"},
          {label: "Median absolute change", value: percent(summary.median_absolute_percentage_change), detail: `${comparisonName} / ${referenceName}`},
          {label: "P90 absolute change", value: percent(summary.p90_absolute_percentage_change), detail: `${comparisonName} / ${referenceName}`},
          {label: "Exposure changing ≥10%", value: percent(summary.weight_share_change_ge_10pct), detail: DATA.metadata.semantics.volume},
          {label: "Exposure moving ≥2 bins", value: percent(summary.weight_share_moved_ge_2_bins), detail: "weighted rank migration"}
        ]);
        summaryFacts($("movement-inspector"), [
          ["Reference model", referenceName],
          ["Comparison model", comparisonName],
          ["View", movementView === "rank" ? "Rank migration" : "Prediction levels"],
          ["Safe aggregate cells", number(source.cells.length, 0)],
          ["Suppressed exposure", percent(source.suppressed_weight_share, 2)]
        ], source.suppressed_weight_share > 0
          ? `Cells below ${DATA.metadata.minimum_cell_size} comparison units were omitted before serialization.`
          : `Every displayed cell contains at least ${DATA.metadata.minimum_cell_size} comparison units.`
        );
      };
      const repairMovementPair = changed => {
        const reference = $("movement-reference");
        const comparison = $("movement-comparison");
        if (reference.value === comparison.value) {
          const other = DATA.models.find(name => name !== (changed === "reference" ? reference.value : comparison.value));
          if (changed === "reference") comparison.value = other; else reference.value = other;
        }
        drawMovement();
      };
      $("movement-reference").addEventListener("change", () => repairMovementPair("reference"));
      $("movement-comparison").addEventListener("change", () => repairMovementPair("comparison"));
      $("movement-view").querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
        movementView = button.dataset.view;
        $("movement-view").querySelectorAll("button").forEach(item => item.classList.toggle("active", item === button));
        drawMovement();
      }));
      $("distribution-view").querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
        hideMovementTooltip();
        reportView = button.dataset.view;
        $("distribution-view").querySelectorAll("button").forEach(item => item.classList.toggle("active", item === button));
        const movementSelected = reportView === "movement";
        $("distribution-density-controls").hidden = movementSelected;
        $("movement-controls").hidden = !movementSelected;
        $("distribution-density-content").hidden = movementSelected;
        $("movement-empty").hidden = !movementSelected || movementAvailable;
        $("movement-content").hidden = !movementSelected || !movementAvailable;
        if (movementSelected && movementAvailable) drawMovement(); else resizePlots($("distribution-density-content"));
        preparePrintPages();
      }));
      draw();
    }

    function renderCurves() {
      let mode = "lorenz";
      let selectedNames = [...DATA.models];
      const draw = () => {
        if (!selectedNames.length) {
          $("curve-chart").replaceChildren(element("div", {class: "review-empty"}, "No models selected"));
          $("curve-legend").replaceChildren();
          $("curve-metrics").replaceChildren();
          $("curve-table").replaceChildren(element("div", {class: "review-empty"}, "Select one or more models to compare discrimination curves."));
          return;
        }
        const series = selectedNames.map(name => ({
          name, x: DATA.curves.models[name][mode].x, y: DATA.curves.models[name][mode].y,
          color: color(name), noPoints: true
        }));
        const benchmark = DATA.curves.benchmark[mode];
        series.push({name: "perfect ordering", x: benchmark.x, y: benchmark.y, color: "#bf6a02", dash: "4 3", noPoints: true});
        series.push({name: "equality", x: [0, 1], y: [0, 1], color: "#8c959f", dash: "7 5", noPoints: true});
        renderLegend($("curve-legend"), series);
        renderXY($("curve-chart"), series, {
          title: mode === "lorenz" ? "Lorenz concentration curve" : "Cumulative gains curve",
          xLabel: DATA.metadata.semantics.curve_x,
          yLabel: DATA.metadata.semantics.curve_y,
          xMin: 0, xMax: 1, yMin: 0, yMax: 1, hideLegend: true
        });
        $("curve-order-chip").textContent = mode === "lorenz" ? "low to high risk" : "high to low risk";
        const metricRows = selectedNames.map(name => DATA.metrics.find(row => row.model === name)).filter(Boolean);
        renderMetricStrip($("curve-metrics"), metricRows.map(row => ({label: row.model, value: number(row.normalized_gini, 4), detail: `raw Gini ${number(row.gini, 4)}`})));
        renderTable($("curve-table"), [
          {key: "model", label: "Model"},
          {key: "gini", label: "Gini", format: value => number(value, 4)},
          {key: "normalized_gini", label: "Normalised", format: value => number(value, 4)}
        ], [...metricRows, {model: "Perfect ordering", gini: DATA.curves.benchmark.gini, normalized_gini: 1}]);
      };
      modelPicker($("curve-model-picker"), $("curve-model-options"), $("curve-model-summary"), DATA.models, names => {
        selectedNames = names; draw();
      });
      $("curve-mode").querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
        mode = button.dataset.mode;
        $("curve-mode").querySelectorAll("button").forEach(item => item.classList.toggle("active", item === button));
        draw();
      }));
      draw();
    }

    function renderDoubleLift() {
      const available = DATA.models.length >= 2;
      $("double-lift-empty").hidden = available; $("double-lift-content").hidden = !available;
      if (!available) return;
      fillSelect($("lift-numerator"), DATA.models, {selected: DATA.models[0]});
      fillSelect($("lift-denominator"), DATA.models, {selected: DATA.models.at(-1)});
      let selectedNames = [...DATA.models];
      const reference = $("lift-reference");
      $("lift-cell-note").textContent = `Bins rank the row-level numerator / denominator and are coarsened until every displayed cell contains at least ${DATA.metadata.minimum_cell_size} distinct comparison units. Actual and model rates are calculated as sum(weight × value) / sum(weight); aggregate ratios divide those sums. Rebasing changes display only.`;
      for (const name of DATA.models) reference.appendChild(element("option", {value: name}, name));
      const draw = () => {
        const numerator = $("lift-numerator").value;
        const denominator = $("lift-denominator").value;
        const entry = DATA.double_lift[numerator][denominator];
        const referenceName = reference.value;
        const divisor = row => referenceName === "none" ? 1 : referenceName === "actual" ? Number(row.actual) : Number(row.predictions[referenceName]);
        const rebase = (value, row) => {
          const base = divisor(row); return finite(value) && finite(base) && base !== 0 ? Number(value) / base : null;
        };
        const x = entry.bins.map(row => row.bin);
        const labels = x.map(String);
        const series = [
          {name: "actual", x, labels, y: entry.bins.map(row => rebase(row.actual, row)), color: "#24292f", actual: true},
          ...selectedNames.map(name => ({name, x, labels, y: entry.bins.map(row => rebase(row.predictions[name], row)), color: color(name)}))
        ];
        renderLegend($("lift-legend"), series);
        renderXY($("lift-chart"), series, {
          title: `Bins ordered by ${numerator} / ${denominator}`,
          xLabel: "weighted double-lift bin (low to high ratio)",
          yLabel: referenceName === "none" ? DATA.metadata.semantics.response : `ratio to ${referenceName === "actual" ? "actual" : referenceName}`,
          baseline: referenceName === "none" ? null : 1,
          categorical: true,
          exposure: {x, labels, y: entry.bins.map(row => row.weight)},
          exposureLabel: DATA.metadata.semantics.volume,
          hideLegend: true
        });
        const comparison = entry.comparison;
        const primaryExact = comparison.primary_score === "exact_nll";
        const scoreName = primaryExact ? "exact NLL" : "mean deviance";
        const interval = finite(comparison.interval_lower) && finite(comparison.interval_upper)
          ? `${number(comparison.interval_lower, 6)} to ${number(comparison.interval_upper, 6)}`
          : "Disabled";
        const calibration = comparison.binned_calibration;
        const reduction = finite(comparison.relative_score_reduction)
          ? percent(comparison.relative_score_reduction, 2)
          : number(Math.abs(primaryExact ? comparison.exact_nll_advantage : comparison.deviance_advantage), 6);
        const reductionDetail = finite(comparison.relative_score_reduction)
          ? `${comparison.lower_score_model} lower than ${comparison.higher_score_model}`
          : `${comparison.lower_score_model} has the lower ${scoreName}`;
        renderMetricStrip($("lift-metrics"), [
          {label: "Comparison", value: `${numerator} vs ${denominator}`, detail: "numerator → denominator"},
          {label: "Paired decision", value: comparison.decision, detail: primaryExact ? "exact held-out NLL" : "held-out mean deviance"},
          {label: "Score reduction", value: reduction, detail: reductionDetail},
          {label: "Line agreement", value: `${number(calibration[numerator].line_agreement, 3)} → ${number(calibration[denominator].line_agreement, 3)}`, detail: "0 = none · 1 = actual line"},
          {label: "Bins", value: String(entry.bins.length), detail: `at least ${DATA.metadata.minimum_cell_size} comparison units per cell`}
        ]);
        const facts = [
          [`${numerator} mean deviance`, number(comparison.mean_deviance[numerator], 6)],
          [`${denominator} mean deviance`, number(comparison.mean_deviance[denominator], 6)],
          ["Deviance score difference", number(comparison.deviance_advantage, 6)],
          ["Paired 95% score difference (denominator − numerator)", interval],
          [`${numerator} binned D²`, number(calibration[numerator].d_squared, 4)],
          [`${denominator} binned D²`, number(calibration[denominator].d_squared, 4)],
          [`${numerator} signed concordance`, number(calibration[numerator].signed_concordance, 4)],
          [`${denominator} signed concordance`, number(calibration[denominator].signed_concordance, 4)],
          [`${numerator} line agreement`, number(calibration[numerator].line_agreement, 4)],
          [`${denominator} line agreement`, number(calibration[denominator].line_agreement, 4)]
        ];
        if (primaryExact) {
          facts.splice(2, 0,
            [`${numerator} exact NLL`, number(comparison.mean_exact_nll[numerator], 6)],
            [`${denominator} exact NLL`, number(comparison.mean_exact_nll[denominator], 6)],
            ["Exact-NLL score difference", number(comparison.exact_nll_advantage, 6)]
          );
        }
        summaryFacts($("lift-comparison-summary"), facts,
          primaryExact
            ? "Exact held-out NLL is primary and uses each model's training-fitted power and dispersion. Line agreement is clipped weighted concordance of each displayed model line with actuals: 0 means no positive agreement and 1 means an exact match."
            : "Exact likelihood metadata is incomplete, so common-power deviance is primary. Line agreement is clipped weighted concordance of each displayed model line with actuals: 0 means no positive agreement and 1 means an exact match."
        );
        const columns = [
          {key: "bin", label: "Bin"},
          {key: "rows", label: "Rows", format: value => number(value, 0)},
          {key: "comparison_units", label: "Units", format: value => number(value, 0)},
          {key: "weight", label: DATA.metadata.semantics.volume, format: value => number(value, 3)},
          {key: "weight_share", label: `${DATA.metadata.semantics.volume} %`, format: value => percent(value)},
          {key: "aggregate_prediction_ratio", label: "Aggregate ratio", format: value => number(value, 4)},
          {key: "actual", label: "Actual", format: value => number(value, 5)},
          {key: "deviance_advantage_contribution", label: "Deviance score Δ", format: value => number(value, 6)}
        ];
        if (primaryExact) columns.push({key: "exact_nll_advantage_contribution", label: "Exact-NLL score Δ", format: value => number(value, 6)});
        renderTable($("lift-table"), columns, entry.bins);
      };
      const repairPair = changed => {
        const numerator = $("lift-numerator"), denominator = $("lift-denominator");
        if (numerator.value === denominator.value) {
          const other = DATA.models.find(name => name !== (changed === "numerator" ? numerator.value : denominator.value));
          if (changed === "numerator") denominator.value = other; else numerator.value = other;
        }
        draw();
      };
      $("lift-numerator").addEventListener("change", () => repairPair("numerator"));
      $("lift-denominator").addEventListener("change", () => repairPair("denominator"));
      modelPicker($("lift-model-picker"), $("lift-model-options"), $("lift-model-summary"), DATA.models, names => {
        selectedNames = names; draw();
      });
      reference.addEventListener("change", draw); draw();
    }

    const PRINT_CHART_HEIGHT = 500;
    const PRINT_COMPARISON_HEIGHT = 462;
    const PRINT_CONTENT_WIDTH = 279 * 96 / 25.4;

    function preparePrintPages() {
      document.querySelectorAll(".print-page-furniture").forEach(node => node.remove());
      const allPages = [...document.querySelectorAll("[data-print-page]")];
      allPages.forEach(page => page.classList.remove("print-last-page"));
      const pages = allPages.filter(page => page.closest("[hidden]") === null);
      const template = $("print-page-furniture-template");
      const generated = String(DATA.metadata.generated_utc || "")
        .replace("T", " ")
        .replace("+00:00", " UTC");
      const meta = `${DATA.metadata.problem_type} · ${Number(DATA.metadata.rows_used).toLocaleString()} rows · Generated ${generated}`;
      pages.forEach((page, index) => {
        const section = page.dataset.printSection || "Model review";
        const fragment = template.content.cloneNode(true);
        const header = fragment.querySelector(".print-page-header");
        const footer = fragment.querySelector(".print-page-footer");
        header.querySelector("[data-print-report-title]").textContent = DATA.metadata.title;
        header.querySelector("[data-print-header-section]").textContent = section;
        header.querySelector("[data-print-meta]").textContent = meta;
        footer.querySelector("[data-print-footer-section]").textContent = section;
        footer.querySelector("[data-print-page-number]").textContent = `Page ${index + 1} of ${pages.length}`;
        page.prepend(header);
        page.append(footer);
      });
      pages.at(-1)?.classList.add("print-last-page");
      return pages;
    }

    function preparePrintPlots() {
      hideMovementTooltip();
      preparePrintPages();
      document.querySelectorAll(".js-plotly-plot").forEach(plot => {
        if (plot.closest("[hidden]")) return;
        const height = plot.closest(".comparison-chart-shell")
          ? PRINT_COMPARISON_HEIGHT
          : PRINT_CHART_HEIGHT;
        Plotly.relayout(plot, {width: PRINT_CONTENT_WIDTH, height});
      });
    }

    function restoreScreenPlots() {
      resizePlots();
    }

    $("printAction").addEventListener("click", () => window.print());
    window.addEventListener("resize", () => resizePlots());
    window.addEventListener("beforeprint", preparePrintPlots);
    window.addEventListener("afterprint", restoreScreenPlots);
    const printMedia = window.matchMedia("print");
    const handlePrintMedia = event => {
      if (event.matches) preparePrintPlots(); else restoreScreenPlots();
    };
    if (typeof printMedia.addEventListener === "function") {
      printMedia.addEventListener("change", handlePrintMedia);
    } else {
      printMedia.addListener(handlePrintMedia);
    }
    initTabs(); renderOverview(); renderImportance(); renderRelativities(); renderInteractions();
    renderDistribution(); renderCurves(); renderDoubleLift();
    preparePrintPages();
    resizePlots(document.querySelector(".review-view.active"));
  })();
  </script>
</body>
</html>
"""


__all__ = ["render_underwriter_html"]
