const heroSelect = document.getElementById("hero-select");
const modeSelect = document.getElementById("mode-select");
const startDateInput = document.getElementById("start-date");
const endDateInput = document.getElementById("end-date");
const findButton = document.getElementById("find-button");

const heroCard = document.getElementById("hero-card");
const marketWatch = document.getElementById("market-watch");
const matchesPanel = document.getElementById("matches");
const industryChain = document.getElementById("industry-chain");
const headlineRegime = document.getElementById("headline-regime");
const activeMode = document.getElementById("active-mode");
const searchBackend = document.getElementById("search-backend");
const heroRegime = document.getElementById("hero-regime");
const effectiveWindowLabel = document.getElementById("effective-window");

let heroes = [];

function traitPills(traits) {
  return traits.map((trait) => `<span class="pill">${trait}</span>`).join("");
}

function hashString(value) {
  return value.split("").reduce((acc, char) => acc + char.charCodeAt(0), 0);
}

function sparklineValues(label, value) {
  const seed = hashString(`${label}-${value}`);
  return Array.from({ length: 8 }, (_, index) => {
    const wave = Math.sin((seed + index * 13) / 9);
    const drift = Math.cos((seed + index * 7) / 11);
    return Math.round(44 + wave * 18 + drift * 10);
  });
}

function normalizeSeriesToSparkline(values) {
  if (!values.length) {
    return [];
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;

  return values.map((value) => Math.round(((value - min) / span) * 80 + 10));
}

function sparklineSvg(values) {
  if (!values.length) {
    return "";
  }

  const step = 100 / Math.max(values.length - 1, 1);
  const points = values
    .map((point, index) => `${index * step},${100 - point}`)
    .join(" ");

  return `
    <svg class="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <polyline class="sparkline-line" points="${points}"></polyline>
    </svg>
  `;
}

function seriesSparkline(series, fallbackLabel = "fallback") {
  if (!series?.length) {
    return sparklineSvg(sparklineValues(fallbackLabel, fallbackLabel));
  }

  const closeValues = series.map((point) => point.close);
  return sparklineSvg(normalizeSeriesToSparkline(closeValues));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function buildWindowRange(series, window) {
  if (!series?.length || !window) {
    return null;
  }

  const startIndex = series.findIndex((point) => point.date >= window.start_date);
  let endIndex = -1;
  for (let index = series.length - 1; index >= 0; index -= 1) {
    if (series[index].date <= window.end_date) {
      endIndex = index;
      break;
    }
  }

  if (startIndex === -1 || endIndex === -1 || endIndex < startIndex) {
    return null;
  }

  return { startIndex, endIndex };
}

function buildHeroChartModel(series, selectedWindow, effectiveWindow = null) {
  if (!series?.length) {
    return {
      svg: `<div class="hero-chart-empty">No price history available.</div>`,
      points: [],
    };
  }

  const width = 760;
  const height = 240;
  const padding = { top: 14, right: 16, bottom: 26, left: 16 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const closes = series.map((point) => point.close);
  const rawMin = Math.min(...closes);
  const rawMax = Math.max(...closes);
  const buffer = Math.max((rawMax - rawMin) * 0.08, rawMax * 0.01, 1);
  const minClose = rawMin - buffer;
  const maxClose = rawMax + buffer;
  const closeSpan = maxClose - minClose || 1;

  const xForIndex = (index) => {
    if (series.length === 1) {
      return padding.left + innerWidth / 2;
    }
    return padding.left + (index / (series.length - 1)) * innerWidth;
  };

  const yForClose = (close) => (
    padding.top + ((maxClose - close) / closeSpan) * innerHeight
  );

  const points = series.map((point, index) => ({
    ...point,
    index,
    x: xForIndex(index),
    y: yForClose(point.close),
  }));

  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");

  const selectedRange = buildWindowRange(series, selectedWindow);
  const effectiveRange = buildWindowRange(series, effectiveWindow);

  const rangeRect = (range, className) => {
    if (!range) {
      return "";
    }
    const startX = xForIndex(range.startIndex);
    const endX = xForIndex(range.endIndex);
    const rectWidth = Math.max(endX - startX, 2);
    return `
      <rect
        class="${className}"
        x="${startX.toFixed(2)}"
        y="${padding.top}"
        width="${rectWidth.toFixed(2)}"
        height="${innerHeight}"
      ></rect>
    `;
  };

  const boundaryLines = (range, className) => {
    if (!range) {
      return "";
    }
    const lines = [range.startIndex, range.endIndex].map((index) => {
      const x = xForIndex(index);
      return `
        <line
          class="${className}"
          x1="${x.toFixed(2)}"
          y1="${padding.top}"
          x2="${x.toFixed(2)}"
          y2="${(height - padding.bottom).toFixed(2)}"
        ></line>
      `;
    });
    return lines.join("");
  };

  const gridLines = Array.from({ length: 4 }, (_, index) => {
    const y = padding.top + (innerHeight / 3) * index;
    return `
      <line
        class="hero-chart-grid"
        x1="${padding.left}"
        y1="${y.toFixed(2)}"
        x2="${(width - padding.right).toFixed(2)}"
        y2="${y.toFixed(2)}"
      ></line>
    `;
  }).join("");

  const svg = `
    <svg
      id="hero-chart-svg"
      class="hero-chart-svg"
      viewBox="0 0 ${width} ${height}"
      preserveAspectRatio="none"
      aria-label="Hero stock price chart"
    >
      ${gridLines}
      ${rangeRect(selectedRange, "hero-chart-range")}
      ${rangeRect(effectiveRange, "hero-chart-effective-range")}
      ${boundaryLines(selectedRange, "hero-chart-boundary")}
      ${boundaryLines(effectiveRange, "hero-chart-effective-boundary")}
      <path class="hero-chart-line" d="${linePath}"></path>
      <line id="hero-chart-focus-line" class="hero-chart-focus-line" x1="0" y1="${padding.top}" x2="0" y2="${height - padding.bottom}" hidden></line>
      <circle id="hero-chart-focus-dot" class="hero-chart-focus-dot" cx="0" cy="0" r="4" hidden></circle>
      <rect
        id="hero-chart-overlay"
        class="hero-chart-overlay"
        x="${padding.left}"
        y="${padding.top}"
        width="${innerWidth}"
        height="${innerHeight}"
      ></rect>
    </svg>
  `;

  return {
    svg,
    points,
    width,
    height,
    padding,
    innerWidth,
    selectedWindow,
    effectiveWindow,
  };
}

function attachHeroChartInteractions(chartModel) {
  if (!chartModel?.points?.length) {
    return;
  }

  const svg = document.getElementById("hero-chart-svg");
  const overlay = document.getElementById("hero-chart-overlay");
  const tooltip = document.getElementById("hero-chart-tooltip");
  const focusLine = document.getElementById("hero-chart-focus-line");
  const focusDot = document.getElementById("hero-chart-focus-dot");

  if (!svg || !overlay || !tooltip || !focusLine || !focusDot) {
    return;
  }

  const updateTooltip = (event) => {
    const overlayRect = overlay.getBoundingClientRect();
    const relativeX = clamp(event.clientX - overlayRect.left, 0, overlayRect.width);
    const progress = relativeX / Math.max(overlayRect.width, 1);
    const pointIndex = clamp(
      Math.round(progress * (chartModel.points.length - 1)),
      0,
      chartModel.points.length - 1,
    );
    const point = chartModel.points[pointIndex];
    const inSelectedWindow = point.date >= chartModel.selectedWindow.start_date
      && point.date <= chartModel.selectedWindow.end_date;
    const inEffectiveWindow = chartModel.effectiveWindow
      && point.date >= chartModel.effectiveWindow.start_date
      && point.date <= chartModel.effectiveWindow.end_date;

    focusLine.removeAttribute("hidden");
    focusDot.removeAttribute("hidden");
    focusLine.setAttribute("x1", point.x);
    focusLine.setAttribute("x2", point.x);
    focusDot.setAttribute("cx", point.x);
    focusDot.setAttribute("cy", point.y);

    tooltip.hidden = false;
    tooltip.innerHTML = `
      <strong>${point.date}</strong>
      <span>Close: ${point.close.toFixed(2)}</span>
      ${inSelectedWindow ? "<span>Inside selected window</span>" : ""}
      ${inEffectiveWindow ? "<span>Inside encoded model window</span>" : ""}
    `;

    const shell = svg.parentElement;
    const shellRect = shell.getBoundingClientRect();
    const left = clamp(
      event.clientX - shellRect.left + 12,
      8,
      shellRect.width - 168,
    );
    const top = clamp(
      event.clientY - shellRect.top - 72,
      8,
      shellRect.height - 72,
    );
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  };

  const clearTooltip = () => {
    tooltip.hidden = true;
    focusLine.setAttribute("hidden", "hidden");
    focusDot.setAttribute("hidden", "hidden");
  };

  overlay.addEventListener("mousemove", updateTooltip);
  overlay.addEventListener("mouseenter", updateTooltip);
  overlay.addEventListener("mouseleave", clearTooltip);
}

function modeLabel(mode) {
  return mode
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function backendLabel(searchMode) {
  if (searchMode === "vqvae") {
    return "Trained VQ-VAE";
  }
  if (searchMode === "mock") {
    return "Curated Demo Layer";
  }
  return "Loading...";
}

function confidenceTone(score) {
  if (score >= 0.9) {
    return "High";
  }
  if (score >= 0.75) {
    return "Strong";
  }
  if (score >= 0.6) {
    return "Moderate";
  }
  return "Weak";
}

function setLoadingState(isLoading) {
  findButton.disabled = isLoading;
  findButton.textContent = isLoading ? "Scanning..." : "Find Mirrors";
}

function renderSummary(hero, mode, matchData) {
  const dna = hero[mode];
  const effectiveWindow = matchData.effective_hero_window;
  const selectedWindow = matchData.selected_window;

  activeMode.textContent = modeLabel(mode);
  searchBackend.textContent = backendLabel(matchData.search_backend);
  heroRegime.textContent = matchData.hero_regime_code || dna.regime_code;
  effectiveWindowLabel.textContent = effectiveWindow
    ? `${effectiveWindow.start_date} to ${effectiveWindow.end_date}`
    : `${selectedWindow.start_date} to ${selectedWindow.end_date}`;
}

function renderHero(
  hero,
  mode,
  selectedWindow,
  effectiveWindow = null,
  searchMode = null,
  heroSeries = null,
) {
  const dna = hero[mode];
  const requestedLabel = `${selectedWindow.start_date} to ${selectedWindow.end_date}`;
  const encodedLabel = effectiveWindow
    ? `${effectiveWindow.start_date} to ${effectiveWindow.end_date} (${effectiveWindow.window_size} trading-day model window)`
    : requestedLabel;
  const backendNote = searchMode === "vqvae"
    ? `<p class="meta">Using trained VQ-VAE Price DNA encoding.</p>`
    : searchMode === "mock"
      ? `<p class="meta">Using curated demo matches for this mode.</p>`
      : "";
  const heroChart = buildHeroChartModel(heroSeries?.series || [], selectedWindow, effectiveWindow);
  const historyLabel = heroSeries?.available_start_date && heroSeries?.available_end_date
    ? `${heroSeries.available_start_date} to ${heroSeries.available_end_date}`
    : requestedLabel;

  heroCard.innerHTML = `
    <div class="card hero-card">
      <h3>${hero.name} (${hero.ticker})</h3>
      <div class="hero-chart-shell">
        ${heroChart.svg}
        <div id="hero-chart-tooltip" class="hero-chart-tooltip" hidden></div>
      </div>
      <div class="hero-chart-legend">
        <span><i class="legend-swatch legend-swatch-selected"></i>Selected window</span>
        ${effectiveWindow ? '<span><i class="legend-swatch legend-swatch-effective"></i>Encoded model window</span>' : ""}
      </div>
      <p class="meta">Selected window: ${requestedLabel}</p>
      <p class="meta">Available history: ${historyLabel}</p>
      <p>${hero.summary}</p>
      <p><strong>${hero.window_label}</strong></p>
      <div class="metric-row">
        <div>
          <span class="mini-label">Confidence</span>
          <p class="score">${Math.round(dna.confidence * 100)}%</p>
        </div>
        <div>
          <span class="mini-label">Regime Code</span>
          <p class="metric-value">${dna.regime_code}</p>
        </div>
      </div>
      <p class="meta">Encoded slice: ${encodedLabel}</p>
      ${backendNote}
      <div>${traitPills(dna.traits)}</div>
    </div>
  `;

  attachHeroChartInteractions(heroChart);
}

function renderMarketWatch(data) {
  headlineRegime.textContent = data.headline_regime;
  marketWatch.innerHTML = data.indicators.map((indicator) => {
    const chartSvg = seriesSparkline(indicator.series, `${indicator.name}-${indicator.value}`);
    const changeLabel = typeof indicator.change_pct === "number"
      ? `<p class="meta">Last ${indicator.series?.length || 0} sessions: ${indicator.change_pct >= 0 ? "+" : ""}${indicator.change_pct}%</p>`
      : "";

    return `
      <div class="card watch-card">
        <div class="card-topline">
          <div>
            <h3>${indicator.name}</h3>
            ${indicator.symbol ? `<p class="meta">${indicator.symbol}</p>` : ""}
          </div>
          <span class="status-pill">${indicator.status}</span>
        </div>
        <div class="sparkline-shell">
          ${chartSvg}
        </div>
        <p class="score">${indicator.value}</p>
        ${changeLabel}
        <p>${indicator.insight}</p>
      </div>
    `;
  }).join("");
}

function renderMatches(items) {
  matchesPanel.innerHTML = items.map((item, index) => {
    const scorePct = Math.round(item.score * 100);
    const chartSvg = item.series?.length
      ? seriesSparkline(item.series, `${item.ticker}-${index}`)
      : "";
    const matchedWindow = item.matched_window
      ? `<p class="meta">Matched window: ${item.matched_window.start_date} to ${item.matched_window.end_date}</p>`
      : "";

    return `
      <div class="card match-card">
        <div class="card-topline">
          <div class="match-title">
            <span class="rank-badge">#${index + 1}</span>
            <div>
              <h3>${item.name} (${item.ticker})</h3>
              <p class="meta">${item.sector}</p>
            </div>
          </div>
          <div class="score-badge">
            <span class="mini-label">${confidenceTone(item.score)}</span>
            <strong>${scorePct}%</strong>
          </div>
        </div>
        <p><strong>${item.regime_label}</strong></p>
        ${matchedWindow}
        ${chartSvg ? `
          <div class="sparkline-shell">
            ${chartSvg}
          </div>
        ` : ""}
        <div class="score-track" aria-hidden="true">
          <span class="score-fill" style="width: ${Math.max(8, Math.min(scorePct, 100))}%"></span>
        </div>
        <p>${item.explanation}</p>
      </div>
    `;
  }).join("");
}

function renderIndustryChain(ticker, relationships) {
  industryChain.innerHTML = `
    <div class="card industry-card">
      <h3>${ticker} industry map</h3>
      <div class="network-grid" aria-hidden="true">
        ${relationships.map((item, index) => `
          <span class="network-node node-${(index % 4) + 1}"></span>
        `).join("")}
      </div>
      <div class="chain-list">
        ${relationships.map((item) => `
          <article class="chain-item">
            <div class="card-topline">
              <strong>${item.ticker}</strong>
              <span class="status-pill">${item.direction}</span>
            </div>
            <p class="meta">${item.relationship}</p>
            <p>${item.impact}</p>
          </article>
        `).join("")}
      </div>
    </div>
  `;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function loadHeroes() {
  const data = await fetchJson("/api/heroes");
  heroes = data.heroes;
  heroSelect.innerHTML = heroes.map((hero) => `
    <option value="${hero.ticker}">${hero.name} (${hero.ticker})</option>
  `).join("");
  syncWindowInputs();
}

function syncWindowInputs() {
  const hero = heroes.find((item) => item.ticker === heroSelect.value) || heroes[0];
  if (!hero) {
    return;
  }
  startDateInput.value = hero.start_date;
  endDateInput.value = hero.end_date;
}

async function loadDashboard() {
  const ticker = heroSelect.value;
  const mode = modeSelect.value;
  const hero = heroes.find((item) => item.ticker === ticker);
  const startDate = startDateInput.value || hero.start_date;
  const endDate = endDateInput.value || hero.end_date;

  setLoadingState(true);

  try {
    const [matchData, chainData, heroSeries] = await Promise.all([
      fetchJson(`/api/mirrors?ticker=${ticker}&mode=${mode}&start_date=${startDate}&end_date=${endDate}`),
      fetchJson(`/api/industry-chain/${ticker}`),
      fetchJson(`/api/price-series?ticker=${ticker}&start_date=${startDate}&end_date=${endDate}`),
    ]);

    renderSummary(hero, mode, matchData);
    renderHero(
      hero,
      mode,
      matchData.selected_window,
      matchData.effective_hero_window,
      matchData.search_backend,
      heroSeries,
    );
    renderMatches(matchData.matches);
    renderIndustryChain(ticker, chainData.relationships);
  } finally {
    setLoadingState(false);
  }
}

async function init() {
  await loadHeroes();
  const watchData = await fetchJson("/api/market-watch");
  renderMarketWatch(watchData);
  await loadDashboard();
}

findButton.addEventListener("click", loadDashboard);
modeSelect.addEventListener("change", loadDashboard);
heroSelect.addEventListener("change", () => {
  syncWindowInputs();
  loadDashboard();
});

init().catch((error) => {
  console.error(error);
  matchesPanel.innerHTML = `<div class="card"><p>Failed to load demo data.</p></div>`;
});
