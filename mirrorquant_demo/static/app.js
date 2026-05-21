const heroSelect = document.getElementById("hero-select");
const modeSelect = document.getElementById("mode-select");
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

function sparklineSvg(values) {
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

  activeMode.textContent = modeLabel(mode);
  searchBackend.textContent = backendLabel(matchData.search_backend);
  heroRegime.textContent = matchData.hero_regime_code || dna.regime_code;
  effectiveWindowLabel.textContent = effectiveWindow
    ? `${effectiveWindow.start_date} to ${effectiveWindow.end_date}`
    : `${hero.start_date} to ${hero.end_date}`;
}

function renderHero(hero, mode, effectiveWindow = null, searchMode = null) {
  const dna = hero[mode];
  const windowLabel = effectiveWindow
    ? `${effectiveWindow.start_date} to ${effectiveWindow.end_date} (${effectiveWindow.window_size} trading-day model window)`
    : `${hero.start_date} to ${hero.end_date}`;
  const backendNote = searchMode === "vqvae"
    ? `<p class="meta">Using trained VQ-VAE Price DNA encoding.</p>`
    : searchMode === "mock"
      ? `<p class="meta">Using curated demo matches for this mode.</p>`
      : "";

  heroCard.innerHTML = `
    <div class="card hero-card">
      <h3>${hero.name} (${hero.ticker})</h3>
      <p class="meta">${windowLabel}</p>
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
      ${backendNote}
      <div>${traitPills(dna.traits)}</div>
    </div>
  `;
}

function renderMarketWatch(data) {
  headlineRegime.textContent = data.headline_regime;
  marketWatch.innerHTML = data.indicators.map((indicator) => `
    <div class="card watch-card">
      <div class="card-topline">
        <h3>${indicator.name}</h3>
        <span class="status-pill">${indicator.status}</span>
      </div>
      <div class="sparkline-shell">
        ${sparklineSvg(sparklineValues(indicator.name, indicator.value))}
      </div>
      <p class="score">${indicator.value}</p>
      <p>${indicator.insight}</p>
    </div>
  `).join("");
}

function renderMatches(items) {
  matchesPanel.innerHTML = items.map((item, index) => {
    const scorePct = Math.round(item.score * 100);
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
}

async function loadDashboard() {
  const ticker = heroSelect.value;
  const mode = modeSelect.value;
  const hero = heroes.find((item) => item.ticker === ticker);

  setLoadingState(true);

  try {
    const [matchData, chainData] = await Promise.all([
      fetchJson(`/api/mirrors?ticker=${ticker}&mode=${mode}`),
      fetchJson(`/api/industry-chain/${ticker}`),
    ]);

    renderSummary(hero, mode, matchData);
    renderHero(hero, mode, matchData.effective_hero_window, matchData.search_backend);
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
heroSelect.addEventListener("change", loadDashboard);

init().catch((error) => {
  console.error(error);
  matchesPanel.innerHTML = `<div class="card"><p>Failed to load demo data.</p></div>`;
});
