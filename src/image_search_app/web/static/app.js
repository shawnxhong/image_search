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
};

function init() {
  els.maxResults.value = String(config.maxResultsPerList);
  els.thumbSize.value = config.thumbnailSize;
  applyModeState();
  els.mode.addEventListener("change", applyModeState);
  els.searchBtn.addEventListener("click", runSearch);
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
      throw new Error(`Search failed: ${res.status}`);
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
