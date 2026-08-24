"""Report-owned CSS primitives for the standalone underwriter HTML."""

REPORT_BASE_CSS = r"""
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #1f2328;
  --muted: #656d76;
  --border: #d0d7de;
  --border-strong: #8c959f;
  --grid: rgba(140, 149, 159, 0.22);
  --blue: #0969da;
  --blue-soft: #dbeafe;
  --red: #d1242f;
  --orange: #bf6a02;
  --yellow: #f4d35e;
  --yellow-border: #d8a10f;
  --focus: #0969da;
  --shadow: rgba(31, 35, 40, 0.16);
  --surface: var(--panel);
  --surface-subtle: #f6f8fa;
  --radius-sm: 4px;
  --radius-md: 6px;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --inspector-width: 360px;
}

*, *::before, *::after { box-sizing: border-box; }

html {
  min-width: 0;
  min-height: 100%;
  background: var(--bg);
}

body {
  display: flex;
  min-width: 0;
  min-height: 100dvh;
  align-items: stretch;
  justify-content: center;
  margin: 0;
  padding: 12px;
  overflow: auto;
  background: var(--bg);
  color: var(--text);
  font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button, select, input { font: inherit; }

button, select {
  height: 30px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-subtle);
  color: var(--text);
}

select { min-width: 150px; background: var(--surface); }
button:hover { background: #eef2f6; }
button:active { background: #e6edf3; }
button:disabled { cursor: not-allowed; opacity: 0.55; }

button:focus-visible,
select:focus-visible,
input:focus-visible,
[tabindex="0"]:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
}

[hidden] { display: none !important; }

.app-shell {
  position: relative;
  display: grid;
  width: min(100%, 1400px);
  min-width: 0;
  height: calc(100dvh - 24px);
  min-height: calc(100dvh - 24px);
  margin: 0 auto;
}

.app-bar, .context-bar {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: var(--space-2);
}

.app-bar {
  grid-area: tabs;
  justify-content: space-between;
  border-bottom: 1px solid var(--border);
}

.app-tabs, .app-actions {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.app-tabs {
  min-width: 0;
  overflow-x: auto;
}

.app-tab {
  height: 32px;
  border-color: transparent;
  border-bottom-color: transparent;
  border-radius: 4px 4px 0 0;
  background: var(--surface);
  color: var(--muted);
}

.app-tab.active {
  border-color: var(--border);
  border-bottom-color: var(--surface);
  color: var(--text);
  font-weight: 600;
  transform: translateY(1px);
}

.context-bar {
  min-height: 40px;
  flex-wrap: wrap;
  padding: var(--space-1) 0 var(--space-2);
}

.context-chip {
  padding: 3px 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-subtle);
  color: var(--muted);
}

.icon-button {
  display: inline-grid;
  width: 32px;
  place-items: center;
  padding: 0;
}

.toolbar-icon {
  width: 17px;
  height: 17px;
  fill: none;
  stroke: currentColor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 1.8;
}

.chart-shell { position: relative; width: 100%; min-height: 0; }
.chart-shell, .review-chart { user-select: none; -webkit-user-select: none; }

.inspector {
  display: flex;
  width: var(--inspector-width);
  min-width: 320px;
  min-height: 0;
  flex-direction: column;
  padding: 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface);
}

.inspector-head { display: flex; align-items: start; gap: 6px; }
.inspector .sidepanel-tabs { min-width: 0; flex: 1 1 auto; overflow-x: auto; }

.sidepanel-tabs {
  display: flex;
  gap: 2px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.sidepanel-tab {
  padding: 5px 8px 6px;
  border: 0;
  border-bottom: 2px solid transparent;
  border-radius: 0;
  background: transparent;
  font-weight: 600;
}

.sidepanel-tab.active {
  border-bottom-color: var(--orange);
  background: var(--surface);
}

.sidepanel-pane {
  display: flex;
  min-height: 0;
  flex: 1 1 auto;
  flex-direction: column;
}

.summary-frame, .report-frame {
  user-select: text;
  -webkit-user-select: text;
}

.summary-frame {
  min-height: 0;
  max-height: none;
  flex: 1 1 auto;
  overflow-x: hidden;
  overflow-y: auto;
  padding-top: 6px;
  border-top: 1px solid #d8dee4;
}

.summary-frame table { max-width: 100%; font-size: 11px !important; }
.compact-summary { display: grid; gap: 8px; }

.summary-facts {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 5px;
}

.summary-fact {
  min-width: 0;
  padding: 5px 6px;
  border: 1px solid #d8dee4;
  border-radius: var(--radius-sm);
  background: var(--surface);
}

.summary-fact span {
  display: block;
  color: var(--muted);
  font-size: 10px;
  line-height: 1.15;
}

.summary-fact strong {
  display: block;
  margin-top: 1px;
  overflow-wrap: anywhere;
  color: var(--text);
  font-size: 12px;
  line-height: 1.25;
}

.summary-note, .summary-empty {
  color: var(--muted);
  font-size: 11px;
}

.summary-note { margin: 0 0 8px; line-height: 1.35; }
.summary-empty { padding: 12px 0; }

.metrics-strip {
  overflow: hidden;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}

.metric-grid { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0; }

.metric-item {
  display: grid;
  min-width: 104px;
  min-height: 60px;
  flex: 1 1 104px;
  grid-template-rows: 28px 20px auto;
  align-items: start;
  padding: 4px 10px;
}

.metric-divider {
  width: 1px;
  min-height: 46px;
  margin: 4px 2px;
  background: #d8dee4;
}

.metric-item-name, .metric-item-delta {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.2;
}

.metric-item-name { overflow: hidden; }
.metric-item-value {
  align-self: start;
  color: var(--text);
  font-size: 15px;
  font-weight: 600;
  line-height: 1.25;
}

.report-panel {
  display: grid;
  min-height: 0;
  overflow: hidden;
  grid-template-rows: auto minmax(0, 1fr);
  padding: 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}

.report-header {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.report-header h2 { margin: 0 0 3px; font-size: 15px; }
.report-header p, .report-note { margin: 0; color: var(--muted); font-size: 12px; }
.report-frame { display: grid; min-height: 0; overflow: auto; align-content: start; gap: 12px; }
.report-section { display: grid; gap: 6px; }
.report-section h3 { margin: 0; font-size: 13px; }

.report-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 12px;
}

.report-table th, .report-table td {
  padding: 6px 8px;
  border-bottom: 1px solid #d8dee4;
  text-align: right;
  vertical-align: top;
}

.report-table th { color: var(--muted); font-weight: 600; }
.report-table th:first-child, .report-table td:first-child { text-align: left; }

.mode-segments {
  display: inline-flex;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
}

.mode-segments button {
  height: 28px;
  border: 0;
  border-right: 1px solid var(--border);
  border-radius: 0;
  background: var(--surface-subtle);
}

.mode-segments button:last-child { border-right: 0; }
.mode-segments button.active {
  background: var(--surface);
  box-shadow: inset 0 -2px 0 var(--orange);
  font-weight: 600;
}

.model-picker-wrap { display: inline-flex; align-items: center; gap: 6px; }
.model-picker-label { color: var(--text); }
.model-picker { position: relative; }

.model-picker > summary {
  box-sizing: border-box;
  min-width: 190px;
  height: 30px;
  overflow: hidden;
  padding: 6px 28px 6px 9px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  font-size: 12px;
  line-height: 16px;
  list-style: none;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-picker > summary::-webkit-details-marker { display: none; }
.model-picker > summary::after {
  position: absolute;
  top: 8px;
  right: 9px;
  content: "▾";
  color: var(--muted);
  font-size: 10px;
}

.model-picker[open] > summary {
  border-color: var(--blue);
  box-shadow: 0 0 0 1px var(--blue);
}

.model-picker-menu {
  position: absolute;
  z-index: 60;
  top: calc(100% + 4px);
  left: 0;
  width: max(240px, 100%);
  overflow: hidden;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  box-shadow: 0 8px 22px var(--shadow);
}

.model-picker-actions {
  display: flex;
  gap: 5px;
  padding: 6px;
  border-bottom: 1px solid var(--border);
  background: var(--surface-subtle);
}

.model-picker-actions button { height: 25px; padding: 3px 8px; }
.model-picker-options { max-height: 210px; overflow-y: auto; padding: 4px 0; }

.model-picker-option {
  display: flex;
  min-height: 29px;
  align-items: center;
  gap: 7px;
  padding: 5px 9px;
  cursor: pointer;
}

.model-picker-option:hover { background: var(--blue-soft); }
.model-picker-option input { margin: 0; }
.model-picker-option span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

#interaction-level-picker .model-picker-menu { width: max(320px, 100%); }
.interaction-diagnostics { max-width: 100%; margin-top: 10px; overflow-x: auto; }
.interaction-diagnostics .report-table { min-width: 520px; }
.interaction-warnings { margin-top: 10px; }
.interaction-warnings .summary-note { color: var(--red, #cf222e); }

.chart-legend-strip {
  box-sizing: border-box;
  display: flex;
  min-height: 34px;
  align-items: center;
  gap: 14px;
  overflow-x: auto;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
  background: var(--surface-subtle);
  white-space: nowrap;
}

.chart-legend-entry {
  display: inline-flex;
  min-width: 0;
  max-width: 260px;
  flex: 0 0 auto;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.chart-legend-swatch { width: 24px; height: 8px; flex: 0 0 24px; }
.chart-legend-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.review-empty {
  padding: 20px;
  border: 1px dashed var(--border);
  border-radius: var(--radius-md);
  color: var(--muted);
  text-align: center;
}

.review-tooltip {
  position: fixed;
  z-index: 80;
  display: none;
  width: max-content;
  max-width: min(300px, calc(100vw - 16px));
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--text);
  color: var(--surface);
  box-shadow: 0 8px 24px var(--shadow);
  pointer-events: none;
}

.review-tooltip strong, .review-tooltip span { display: block; }
.review-tooltip span { margin-top: 2px; color: #eaeef2; font-size: 12px; }

.movement-hover-tooltip {
  position: fixed;
  z-index: 100;
  width: max-content;
  min-width: 230px;
  max-width: min(340px, calc(100vw - 16px));
  padding: 9px 10px;
  border: 1px solid #64115f;
  border-radius: var(--radius-sm);
  background: #fff7d6;
  color: var(--text);
  box-shadow: 0 8px 24px var(--shadow);
  pointer-events: none;
}

.movement-hover-tooltip[hidden] { display: none !important; }
.movement-hover-title { margin-bottom: 6px; font-size: 12px; font-weight: 650; }
.movement-hover-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  column-gap: 14px;
  row-gap: 3px;
  align-items: baseline;
  font-size: 11px;
}

.movement-hover-label { min-width: 0; overflow-wrap: anywhere; color: var(--muted); }
.movement-hover-value { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }

.print-page-furniture { display: none; }

.axis, .tick { stroke: #8c959f; stroke-width: 1; }
.grid { stroke: var(--grid); stroke-width: 1; }
.zero { stroke: var(--border); stroke-width: 1; stroke-dasharray: 4 4; }
.label { fill: var(--text); font-size: 13px; }
.tick-label, .legend { fill: var(--muted); font-size: 11px; }

@media (max-width: 1250px) {
  .metric-grid { min-height: 152px; }
}

@media print {
  .review-tooltip,
  .movement-hover-tooltip,
  .review-empty { display: none !important; }
  .print-page-furniture { display: flex !important; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    animation-duration: 0.001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
  }
}
"""
