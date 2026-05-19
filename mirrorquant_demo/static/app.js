const heroSelect = document.getElementById("hero-select");
const modeSelect = document.getElementById("mode-select");
const findButton = document.getElementById("find-button");

const heroCard = document.getElementById("hero-card");
const marketWatch = document.getElementById("market-watch");
const matchesPanel = document.getElementById("matches");
const industryChain = document.getElementById("industry-chain");
const headlineRegime = document.getElementById("headline-regime");

let heroes = [];

function traitPills(traits) {
  return traits.map((trait) => `<span class="pill">${trait}</span>`).join("");
}

function renderHero(hero, mode) {
  const dna = hero[mode];
  heroCard.innerHTML = `
    <div class="card">
      <h3>${hero.name} (${hero.ticker})</h3>
      <p class="meta">${hero.start_date} to ${hero.end_date}</p>
      <p>${hero.summary}</p>
      <p><strong>${hero.window_label}</strong></p>
      <p class="score">${Math.round(dna.confidence * 100)}%</p>
      <p class="meta">${dna.regime_code}</p>
      <div>${traitPills(dna.traits)}</div>
    </div>
  `;
}

function renderMarketWatch(data) {
  headlineRegime.textContent = data.headline_regime;
  marketWatch.innerHTML = data.indicators.map((indicator) => `
    <div class="card">
      <h3>${indicator.name}</h3>
      <p class="score">${indicator.value}</p>
      <p><strong>${indicator.status}</strong></p>
      <p>${indicator.insight}</p>
    </div>
  `).join("");
}

function renderMatches(items) {
  matchesPanel.innerHTML = items.map((item) => `
    <div class="card">
      <h3>${item.name} (${item.ticker})</h3>
      <p class="score">${Math.round(item.score * 100)}%</p>
      <p><strong>${item.regime_label}</strong></p>
      <p class="meta">${item.sector}</p>
      <p>${item.explanation}</p>
    </div>
  `).join("");
}

function renderIndustryChain(ticker, relationships) {
  industryChain.innerHTML = `
    <div class="card">
      <h3>${ticker} industry map</h3>
      <ul>
        ${relationships.map((item) => `
          <li>
            <strong>${item.ticker}</strong> · ${item.direction} · ${item.relationship}<br/>
            ${item.impact}
          </li>
        `).join("")}
      </ul>
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

  renderHero(hero, mode);

  const [matchData, chainData] = await Promise.all([
    fetchJson(`/api/mirrors?ticker=${ticker}&mode=${mode}`),
    fetchJson(`/api/industry-chain/${ticker}`),
  ]);

  renderMatches(matchData.matches);
  renderIndustryChain(ticker, chainData.relationships);
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
