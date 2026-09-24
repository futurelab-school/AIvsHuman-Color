// User interface for the color game. The game logic lives in game.py and is
// reached through window.pyNewGame() and window.pySubmit(c, m, y, k), which
// both return the game state as a JSON string.

const $ = (id) => document.getElementById(id);
const channels = ["c", "m", "y", "k"];
let busy = false;

const rgbCss = ([r, g, b]) => `rgb(${r}, ${g}, ${b})`;
const pct = (v) => `${Math.round(v * 100)}%`;

function swatchCard(s, label) {
  const card = document.createElement("div");
  card.className = "card";
  const sw = document.createElement("div");
  sw.className = "swatch";
  sw.style.background = rgbCss(s.rgb);
  const cap = document.createElement("div");
  cap.className = "caption";
  const [c, m, y, k] = s.cmyk.map(pct);
  cap.innerHTML = `${label}<br>C ${c} M ${m}<br>Y ${y} K ${k}` +
    (s.error != null ? `<br><strong>Error ${s.error.toFixed(1)}%</strong>` : "");
  card.append(sw, cap);
  return card;
}

function renderList(el, items, labelPrefix, emptyText) {
  el.replaceChildren();
  if (!items.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = emptyText;
    el.append(p);
    return;
  }
  items.forEach((s, i) => el.append(swatchCard(s, `${labelPrefix} ${i + 1}`)));
}

function render(state) {
  $("target-swatch").style.background = rgbCss(state.target.rgb);
  $("target-rgb").textContent = `RGB ${state.target.rgb.join(", ")}`;
  $("attempts").textContent = state.user_attempts.length;
  $("best").textContent = state.best_error == null ? "–" : `${state.best_error.toFixed(1)}%`;

  renderList($("user-attempts"), state.user_attempts, "#", "No guesses yet.");
  renderList($("ai-attempts"), state.ai_attempts, "#", "The AI guesses when you do.");
  renderList($("random-samples"), state.random_samples, "Sample", "");

  const status = $("status");
  status.className = "status";
  if (state.winner === "user") {
    status.textContent = "You win! 💪🧑";
    status.classList.add("win");
  } else if (state.winner === "ai") {
    status.textContent = "AI wins! 🦾🤖";
    status.classList.add("lose");
  } else {
    status.textContent = `Goal: < ${state.win_threshold}% error before the AI`;
  }
  $("submit").disabled = !!state.winner;
}

// Python runs on the page's main thread, so let the browser paint the
// "thinking" message before starting a slow call.
const nextPaint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function run(label, fn) {
  if (busy) return;
  busy = true;
  $("submit").disabled = true;
  $("new-game").disabled = true;
  $("status").className = "status";
  $("status").textContent = label;
  await nextPaint();
  try {
    render(JSON.parse(fn()));
  } catch (err) {
    console.error(err);
    $("status").textContent = "Something went wrong. Check the console.";
    $("submit").disabled = false;
  } finally {
    busy = false;
    $("new-game").disabled = false;
  }
}

function start() {
  $("loading").hidden = true;
  $("game").hidden = false;
  run("Setting up…", () => window.pyNewGame());
}

for (const ch of channels) {
  $(ch).addEventListener("input", () => { $(`${ch}-out`).textContent = `${$(ch).value}%`; });
}

$("controls").addEventListener("submit", (e) => {
  e.preventDefault();
  const [c, m, y, k] = channels.map((ch) => Number($(ch).value) / 100);
  run("AI is thinking…", () => window.pySubmit(c, m, y, k));
});

$("new-game").addEventListener("click", () => {
  if (confirm("Start a new game with a new target color?")) run("Setting up…", () => window.pyNewGame());
});

if (window.pyNewGame) start();
else window.addEventListener("python-ready", start, { once: true });
