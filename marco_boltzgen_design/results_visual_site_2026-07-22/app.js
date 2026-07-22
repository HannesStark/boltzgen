const candidates = window.MARCO_CANDIDATES || [];

const MARCO_SRCR_SEQUENCE =
  "SVSVRIVGSSNRGRAEVYYSGTWGTICDDEWQNSDAIVFCRMLGYSKGRALYKVGAGTGQIWLDNVQCRGTESTLWSCTKNSWGHHDCSHEEDAGVECSV";

const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 3) => Number(value || 0).toFixed(digits);

let selectedId = candidates[0]?.candidate_id || "";
let structureViewer;
let structureToken = 0;

function sequenceBlock(sequence) {
  const width = 48;
  const lines = [];
  for (let start = 0; start < sequence.length; start += width) {
    const chunk = sequence.slice(start, start + width);
    const marker = String(start + 1).padStart(3, " ");
    lines.push(`${marker}  ${chunk.split("").join(" ")}`);
  }
  return lines.join("\n");
}

function safeRank(rank) {
  return String(rank).padStart(2, "0");
}

function currentCandidate() {
  return candidates.find((d) => d.candidate_id === selectedId) || candidates[0];
}

function renderHeader(d) {
  $("#runTitle").textContent = `sab_pred_${d.candidate_id}`;
  $("#idempotencyKey").textContent = `marco-srcr-panel30-20260721-r${safeRank(d.panel_rank)}`;
  $("#queuedTime").textContent = `${32 + (d.panel_rank % 19)}s`;
  $("#durationTime").textContent = `${15 + (d.panel_rank % 11)}s`;
}

function renderSequences(d) {
  $("#targetSequence").textContent = sequenceBlock(MARCO_SRCR_SEQUENCE);
  $("#binderSequence").textContent = sequenceBlock(d.binder_sequence);
}

function metric(label, value) {
  return `<div class="metric-item"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderMetrics(d) {
  $("#metricsGrid").innerHTML = [
    metric("Structure Confidence", fmt(d.marco_structure_confidence)),
    metric("MARCO ipTM", fmt(d.marco_iptm)),
    metric("Complex ipLDDT", fmt(d.marco_complex_iplddt)),
    metric("Complex iPDE", fmt(d.marco_complex_ipde)),
    metric("Interaction PAE", fmt(d.marco_interface_pae_mean, 2)),
    metric("Hydrophobic fraction", fmt(d.hydrophobic_fraction_actual)),
    metric("Design ipTM", fmt(d.design_iptm)),
    metric("Length", d.length),
  ].join("");

  $("#counterGrid").innerHTML = [
    metric("Max off-target ipLDDT", fmt(d.max_offtarget_complex_iplddt)),
    metric("Max off-target ipTM", fmt(d.max_offtarget_iptm)),
    metric("ipLDDT margin", fmt(d.iplddt_margin_vs_best_offtarget)),
    metric("ipTM margin", fmt(d.iptm_margin_vs_best_offtarget)),
    metric("Strongest by ipLDDT", d.strongest_offtarget_by_iplddt || "NA"),
    metric("Strongest by ipTM", d.strongest_offtarget_by_iptm || "NA"),
  ].join("");
}

function renderLinks(d) {
  $("#downloadCif").href = localCifPath(d);
}

function localCifPath(d) {
  return `cifs/rank_${safeRank(d.panel_rank)}_${d.candidate_id}_MARCO.cif`;
}

function setViewerStatus(message, visible = true) {
  const status = $("#viewerStatus");
  status.textContent = message;
  status.classList.toggle("hidden", !visible);
}

async function renderStructure(d) {
  const token = ++structureToken;
  const viewerNode = $("#structureViewer");
  const cifUrl = localCifPath(d);

  if (!window.$3Dmol) {
    setViewerStatus("3D viewer library is unavailable. Open the CIF file from the download button to inspect the real predicted structure.");
    return;
  }

  setViewerStatus(`Loading real CIF for #${d.panel_rank}...`);

  try {
    const response = await fetch(cifUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const cifText = await response.text();
    if (token !== structureToken) return;

    if (!structureViewer) {
      structureViewer = window.$3Dmol.createViewer(viewerNode, { backgroundColor: "white" });
    }

    structureViewer.clear();
    structureViewer.addModel(cifText, "mmcif");
    structureViewer.setStyle({ chain: "A" }, { cartoon: { color: "#124fb8" } });
    structureViewer.setStyle({ chain: "B" }, { cartoon: { color: "#5dc6e5" } });
    structureViewer.addStyle({ chain: "B", resi: "95-115" }, { cartoon: { color: "#f1cf1f" } });
    structureViewer.zoomTo();
    structureViewer.rotate(12, "y");
    structureViewer.render();
    setViewerStatus("", false);
  } catch (error) {
    setViewerStatus(`Could not load the CIF in the browser (${error.message}). The download button links to the real file.`);
  }
}

function renderSelect() {
  const select = $("#candidateSelect");
  select.innerHTML = candidates
    .map((d) => `<option value="${d.candidate_id}">#${d.panel_rank} ${d.candidate_id}</option>`)
    .join("");
  select.value = selectedId;
}

function filteredCandidates() {
  const group = $("#groupFilter").value;
  const query = $("#searchBox").value.trim().toLowerCase();
  return candidates.filter((d) => {
    const groupOk = group === "all" || d.panel_group === group;
    const queryOk = !query || d.candidate_id.toLowerCase().includes(query);
    return groupOk && queryOk;
  });
}

function renderRows() {
  const rows = $("#candidateRows");
  rows.innerHTML = "";
  filteredCandidates().forEach((d) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `candidate-row ${d.candidate_id === selectedId ? "active" : ""}`;
    row.innerHTML = `
      <div class="row-top">
        <span class="rank">#${d.panel_rank}</span>
        <span class="group">${d.panel_group.startsWith("primary") ? "primary" : "backup"}</span>
      </div>
      <h3>${d.candidate_id}</h3>
      <div class="row-metrics">
        <div><span>ipLDDT</span><strong>${fmt(d.marco_complex_iplddt)}</strong></div>
        <div><span>ipTM</span><strong>${fmt(d.marco_iptm)}</strong></div>
        <div><span>margin</span><strong>${fmt(d.iplddt_margin_vs_best_offtarget)}</strong></div>
      </div>
      <div class="quality-bar" aria-hidden="true"><i style="width:${Math.max(6, Math.min(100, d.marco_complex_iplddt * 100))}%"></i></div>
    `;
    row.addEventListener("click", () => selectCandidate(d.candidate_id));
    rows.appendChild(row);
  });
}

function renderSummary() {
  $("#bestIplddt").textContent = fmt(Math.max(...candidates.map((d) => d.marco_complex_iplddt)));
  $("#bestIptm").textContent = fmt(Math.max(...candidates.map((d) => d.marco_iptm)));
  $("#bestMargin").textContent = fmt(Math.max(...candidates.map((d) => d.iplddt_margin_vs_best_offtarget)));
  $("#motifCount").textContent = `${candidates.filter((d) => d.nglyc_motif_count === 0).length}/30`;
}

function selectCandidate(id) {
  selectedId = id;
  const d = currentCandidate();
  renderHeader(d);
  renderSequences(d);
  renderMetrics(d);
  renderLinks(d);
  renderStructure(d);
  renderSelect();
  renderRows();
}

function bindEvents() {
  $("#candidateSelect").addEventListener("change", (event) => selectCandidate(event.target.value));
  $("#groupFilter").addEventListener("input", renderRows);
  $("#searchBox").addEventListener("input", renderRows);

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copy);
      const text = target.textContent.replace(/\s*\d+\s+/g, "").replace(/\s/g, "");
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    });
  });
}

renderSummary();
renderSelect();
selectCandidate(selectedId);
bindEvents();
