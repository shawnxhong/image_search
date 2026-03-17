const config = window.APP_CONFIG;

const els = {
  mode: document.getElementById("queryMode"),
  textQuery: document.getElementById("textQuery"),
  imagePath: document.getElementById("imagePath"),
  maxResults: document.getElementById("maxResults"),
  thumbSize: document.getElementById("thumbSize"),
  searchBtn: document.getElementById("searchBtn"),
  status: document.getElementById("status"),
  solidResults: document.getElementById("solidResults"),
  softResults: document.getElementById("softResults"),
  solidCount: document.getElementById("solidCount"),
  softCount: document.getElementById("softCount"),
  // LLM controls
  llmModel: document.getElementById("llmModel"),
  llmDevice: document.getElementById("llmDevice"),
  llmLoadBtn: document.getElementById("llmLoadBtn"),
  llmUnloadBtn: document.getElementById("llmUnloadBtn"),
  llmIndicator: document.getElementById("llmIndicator"),
  llmStatusText: document.getElementById("llmStatusText"),
};

// ── LLM Control ──────────────────────────────────────────────

async function fetchLLMStatus() {
  try {
    const res = await fetch("/llm/status");
    if (!res.ok) throw new Error("Failed to fetch LLM status");
    return await res.json();
  } catch {
    return { loaded: false };
  }
}

async function fetchLLMAvailable() {
  try {
    const res = await fetch("/llm/available");
    if (!res.ok) throw new Error("Failed to fetch available models");
    return await res.json();
  } catch {
    return { models: [], devices: ["CPU", "GPU"] };
  }
}

function updateLLMUI(status) {
  if (status.loaded) {
    els.llmIndicator.className = "llm-indicator on";
    els.llmStatusText.textContent = `Loaded: ${status.model_name} on ${status.device}`;
    els.llmLoadBtn.disabled = true;
    els.llmUnloadBtn.disabled = false;
  } else {
    els.llmIndicator.className = "llm-indicator off";
    els.llmStatusText.textContent = "Not loaded";
    els.llmLoadBtn.disabled = false;
    els.llmUnloadBtn.disabled = true;
  }
}

function setLLMLoading(message) {
  els.llmIndicator.className = "llm-indicator loading";
  els.llmStatusText.textContent = message;
  els.llmLoadBtn.disabled = true;
  els.llmUnloadBtn.disabled = true;
}

async function populateLLMControls() {
  const [status, available] = await Promise.all([
    fetchLLMStatus(),
    fetchLLMAvailable(),
  ]);

  // Populate model dropdown
  els.llmModel.innerHTML = "";
  if (available.models.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No models found";
    els.llmModel.appendChild(opt);
  } else {
    for (const model of available.models) {
      const opt = document.createElement("option");
      opt.value = model;
      opt.textContent = model;
      els.llmModel.appendChild(opt);
    }
  }

  // Populate device dropdown
  els.llmDevice.innerHTML = "";
  for (const device of available.devices) {
    const opt = document.createElement("option");
    opt.value = device;
    opt.textContent = device;
    els.llmDevice.appendChild(opt);
  }

  // Pre-select current model/device if loaded
  if (status.loaded && status.model_name) {
    els.llmModel.value = status.model_name;
    els.llmDevice.value = status.device;
  }

  updateLLMUI(status);
}

async function loadLLM() {
  const modelName = els.llmModel.value;
  const device = els.llmDevice.value;
  if (!modelName) return;

  setLLMLoading(`Loading ${modelName} on ${device}...`);

  try {
    const res = await fetch("/llm/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName, device }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Load failed: ${res.status}`);
    }
    const status = await res.json();
    updateLLMUI(status);
  } catch (error) {
    updateLLMUI({ loaded: false });
    els.llmStatusText.textContent = `Error: ${error.message}`;
  }
}

async function unloadLLM() {
  setLLMLoading("Unloading...");

  try {
    const res = await fetch("/llm/unload", { method: "POST" });
    if (!res.ok) throw new Error(`Unload failed: ${res.status}`);
    const status = await res.json();
    updateLLMUI(status);
  } catch (error) {
    const status = await fetchLLMStatus();
    updateLLMUI(status);
  }
}

// ── Search ───────────────────────────────────────────────────

function init() {
  els.maxResults.value = String(config.maxResultsPerList);
  els.thumbSize.value = config.thumbnailSize;
  applyModeState();
  els.mode.addEventListener("change", applyModeState);
  els.searchBtn.addEventListener("click", runSearch);
  els.textQuery.addEventListener("keydown", (event) => {
    if (event.key === "Enter") runSearch();
  });
  els.imagePath.addEventListener("keydown", (event) => {
    if (event.key === "Enter") runSearch();
  });

  // LLM controls
  els.llmLoadBtn.addEventListener("click", loadLLM);
  els.llmUnloadBtn.addEventListener("click", unloadLLM);
  populateLLMControls();
}

function applyModeState() {
  const mode = els.mode.value;
  els.textQuery.disabled = mode === "image";
  els.imagePath.disabled = mode === "text";
}

function thumbnailPixels() {
  return config.thumbnailPixelBySize[els.thumbSize.value] || config.thumbnailPixelBySize.medium;
}

function setStatus(text) {
  els.status.textContent = text;
}

function createCard(item) {
  const card = document.createElement("article");
  card.className = "card";

  const img = document.createElement("img");
  img.className = "thumb";
  img.style.height = `${thumbnailPixels()}px`;
  img.alt = item.file_path;
  img.src = `/image-preview?path=${encodeURIComponent(item.file_path)}`;
  img.loading = "lazy";

  const content = document.createElement("div");
  content.className = "content";
  content.innerHTML = `
    <div><strong>Score:</strong> ${Number(item.score).toFixed(4)}</div>
    <div class="muted">${item.file_path}</div>
    <p>${item.explanation?.reason || "No explanation"}</p>
  `;

  card.appendChild(img);
  card.appendChild(content);
  return card;
}

function renderSection(container, countEl, items, emptyText) {
  container.innerHTML = "";
  countEl.textContent = String(items.length);
  if (items.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }
  items.forEach((item) => container.appendChild(createCard(item)));
}

async function runSearch() {
  const mode = els.mode.value;
  const topK = Math.max(1, Number(els.maxResults.value || config.maxResultsPerList));
  const textQuery = els.textQuery.value.trim();
  const imagePath = els.imagePath.value.trim();

  if ((mode === "text" || mode === "image+text") && !textQuery) {
    setStatus("Please enter a text query.");
    return;
  }
  if ((mode === "image" || mode === "image+text") && !imagePath) {
    setStatus("Please enter an image path.");
    return;
  }

  setStatus("Searching...");

  try {
    let endpoint = "/search/text";
    let payload = { query: textQuery, top_k: topK };

    if (mode === "image" || mode === "image+text") {
      endpoint = "/search/image";
      payload = {
        image_path: imagePath,
        query: mode === "image+text" ? textQuery : null,
        top_k: topK,
      };
    }

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Search failed: ${res.status}`);
    }

    const data = await res.json();

    // Required behavior: soft = semantic results minus solid (strictly non-overlapping)
    const solidIds = new Set((data.solid_results || []).map((x) => x.image_id));
    const softNonOverlap = (data.soft_results || []).filter((x) => !solidIds.has(x.image_id));

    const cap = topK;
    renderSection(
      els.solidResults,
      els.solidCount,
      (data.solid_results || []).slice(0, cap),
      "No solid matches found.",
    );
    renderSection(
      els.softResults,
      els.softCount,
      softNonOverlap.slice(0, cap),
      "No additional soft matches.",
    );

    setStatus("Search complete.");
  } catch (error) {
    setStatus(error.message || "Search failed.");
  }
}

init();
