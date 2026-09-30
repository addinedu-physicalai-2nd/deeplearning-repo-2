"use strict";

const TABLES = {
  frame_summary: {
    title: "Frame Summary",
    description: "프레임별 객체 검출 개수를 보여줍니다.",
  },
  pothole: {
    title: "Pothole",
    description: "포트홀 위치, 신뢰도와 위험도를 보여줍니다.",
  },
  people: {
    title: "People",
    description: "사람 검출 위치와 신뢰도를 보여줍니다.",
  },
  trafficCone: {
    title: "Traffic Cone",
    description: "교통콘 검출 위치와 신뢰도를 보여줍니다.",
  },
  car: {
    title: "Car",
    description: "차량 검출 위치와 신뢰도를 보여줍니다.",
  },
  lane: {
    title: "Lane",
    description: "차선 클래스와 신뢰도를 보여줍니다.",
  },
};

const COLUMN_LABELS = {
  id: "ID",
  frame_summary_id: "Summary ID",
  frame: "Frame",
  sequence_id: "Sequence ID",
  class_id: "Class ID",
  confidence: "Confidence",
  b_box: "Bounding Box",
  risk_level: "Risk Level",
  pothole_count: "Pothole",
  people_count: "People",
  trafficCone_count: "Traffic Cone",
  car_count: "Car",
  lane_count: "Lane",
  created_at: "Created At",
};

const state = {
  table: "frame_summary",
  page: 1,
  pageSize: 20,
  totalPages: 1,
  frame: "",
  refreshing: false,
};

const body = document.body;
const urls = {
  summary: body.dataset.summaryUrl,
  status: body.dataset.statusUrl,
  table: body.dataset.tableUrl,
  databaseReset: body.dataset.databaseResetUrl,
};

const elements = {
  navigation: document.querySelector("#table-navigation"),
  title: document.querySelector("#table-title"),
  description: document.querySelector("#table-description"),
  tableHead: document.querySelector("#data-table-head"),
  tableBody: document.querySelector("#data-table-body"),
  error: document.querySelector("#dashboard-error"),
  refresh: document.querySelector("#refresh-button"),
  updated: document.querySelector("#last-updated"),
  filterForm: document.querySelector("#filter-form"),
  filterInput: document.querySelector("#frame-filter"),
  databaseReset: document.querySelector("#database-reset"),
  pageSize: document.querySelector("#page-size"),
  previousPage: document.querySelector("#previous-page"),
  nextPage: document.querySelector("#next-page"),
  pageIndicator: document.querySelector("#page-indicator"),
  rowCount: document.querySelector("#row-count"),
  autoRefresh: document.querySelector("#auto-refresh"),
};

function buildTableUrl() {
  const baseUrl = urls.table.replace("TABLE_NAME", encodeURIComponent(state.table));
  const params = new URLSearchParams({
    page: String(state.page),
    page_size: String(state.pageSize),
  });
  if (state.frame !== "") {
    params.set("frame", state.frame);
  }
  return `${baseUrl}?${params.toString()}`;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function setMetric(id, value) {
  document.querySelector(id).textContent = Number(value || 0).toLocaleString("ko-KR");
}

async function loadSummary() {
  const summary = await fetchJson(urls.summary);
  setMetric("#metric-frame", summary.frame_count);
  setMetric("#metric-pothole", summary.pothole_count);
  setMetric("#metric-people", summary.people_count);
  setMetric("#metric-cone", summary.trafficCone_count);
  setMetric("#metric-car", summary.car_count);
  setMetric("#metric-lane", summary.lane_count);
}

function setSystemStatus(name, status, label) {
  const dot = document.querySelector(`#${name}-status-dot`);
  const text = document.querySelector(`#${name}-status-text`);
  dot.classList.remove("is-online", "is-error");
  if (status === "online") {
    dot.classList.add("is-online");
  } else if (status === "error") {
    dot.classList.add("is-error");
  }
  text.textContent = label;
}

async function loadSystemStatus() {
  const status = await fetchJson(urls.status);
  setSystemStatus(
    "udp",
    status.received_frame_id !== null ? "online" : "waiting",
    status.received_frame_id !== null ? `#${status.received_frame_id}` : "대기",
  );
  setSystemStatus(
    "ai",
    status.error ? "error" : status.ai_frame_id !== null ? "online" : "waiting",
    status.error ? "오류" : status.ai_frame_id !== null ? `#${status.ai_frame_id}` : "대기",
  );
  setSystemStatus(
    "db",
    status.db_error ? "error" : status.db_connected ? "online" : "waiting",
    status.db_error ? "오류" : status.db_connected ? "연결" : "대기",
  );
  setSystemStatus(
    "monitoring",
    status.qt_error ? "error" : status.qt_frame_id !== null ? "online" : "waiting",
    status.qt_error ? "오류" : status.qt_frame_id !== null ? `#${status.qt_frame_id}` : "대기",
  );
}

function createRiskBadge(value) {
  const badge = document.createElement("span");
  const normalized = String(value || "unknown").toLowerCase();
  badge.className = `risk-badge risk-${normalized}`;
  badge.textContent = value ?? "-";
  return badge;
}

function createCell(column, value) {
  const cell = document.createElement("td");
  if (column === "risk_level") {
    cell.append(createRiskBadge(value));
    return cell;
  }
  if (column === "b_box") {
    cell.classList.add("cell-monospace");
  }
  if (column === "confidence") {
    cell.classList.add("confidence-value");
  }
  cell.textContent = value ?? "-";
  return cell;
}

function renderTable(payload) {
  elements.tableHead.replaceChildren();
  elements.tableBody.replaceChildren();

  const headerRow = document.createElement("tr");
  payload.columns.forEach((column) => {
    const header = document.createElement("th");
    header.scope = "col";
    header.textContent = COLUMN_LABELS[column] || column;
    headerRow.append(header);
  });
  elements.tableHead.append(headerRow);

  if (payload.rows.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.className = "empty-cell";
    cell.colSpan = Math.max(payload.columns.length, 1);
    cell.textContent = state.frame === ""
      ? "저장된 데이터가 없습니다."
      : `Frame ${state.frame}에 해당하는 데이터가 없습니다.`;
    row.append(cell);
    elements.tableBody.append(row);
  } else {
    payload.rows.forEach((record) => {
      const row = document.createElement("tr");
      payload.columns.forEach((column) => {
        row.append(createCell(column, record[column]));
      });
      elements.tableBody.append(row);
    });
  }

  state.page = payload.page;
  state.totalPages = payload.total_pages;
  elements.rowCount.textContent = `총 ${Number(payload.total).toLocaleString("ko-KR")}개 행`;
  elements.pageIndicator.textContent = `${payload.page} / ${payload.total_pages}`;
  elements.previousPage.disabled = payload.page <= 1;
  elements.nextPage.disabled = payload.page >= payload.total_pages;
}

function showTableLoading() {
  elements.tableBody.replaceChildren();
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.className = "empty-cell";
  cell.colSpan = Math.max(elements.tableHead.querySelectorAll("th").length, 1);
  cell.textContent = "데이터를 불러오는 중입니다.";
  row.append(cell);
  elements.tableBody.append(row);
}

async function loadTable(showLoading = true) {
  if (showLoading) {
    showTableLoading();
  }
  const payload = await fetchJson(buildTableUrl());
  renderTable(payload);
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.classList.remove("d-none");
}

function clearError() {
  elements.error.classList.add("d-none");
  elements.error.textContent = "";
}

async function refreshDashboard(showLoading = false) {
  if (state.refreshing) {
    return;
  }
  state.refreshing = true;
  elements.refresh.disabled = true;
  clearError();

  const results = await Promise.allSettled([
    loadSummary(),
    loadSystemStatus(),
    loadTable(showLoading),
  ]);
  const failures = results.filter((result) => result.status === "rejected");
  if (failures.length > 0) {
    showError(failures[0].reason.message || "데이터를 불러오지 못했습니다.");
  } else {
    elements.updated.textContent = `최근 갱신 ${new Date().toLocaleTimeString("ko-KR")}`;
  }

  elements.refresh.disabled = false;
  state.refreshing = false;
}

function selectTable(tableName) {
  if (!TABLES[tableName]) {
    return;
  }
  state.table = tableName;
  state.page = 1;
  elements.title.textContent = TABLES[tableName].title;
  elements.description.textContent = TABLES[tableName].description;
  document.querySelectorAll("[data-table]").forEach((button) => {
    button.classList.toggle("active", button.dataset.table === tableName);
  });
  refreshDashboard(true);
}

elements.navigation.addEventListener("click", (event) => {
  const button = event.target.closest("[data-table]");
  if (button) {
    selectTable(button.dataset.table);
  }
});

elements.filterForm.addEventListener("submit", (event) => {
  event.preventDefault();
  state.frame = elements.filterInput.value.trim();
  state.page = 1;
  refreshDashboard(true);
});

elements.databaseReset.addEventListener("click", async () => {
  const confirmed = window.confirm(
    "모든 테이블의 검출 데이터를 삭제합니다. 이 작업은 되돌릴 수 없습니다. 계속하시겠습니까?",
  );
  if (!confirmed) {
    return;
  }

  elements.databaseReset.disabled = true;
  clearError();
  try {
    const response = await fetch(urls.databaseReset, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ confirmation: "DELETE_ALL_DATA" }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    elements.filterInput.value = "";
    state.frame = "";
    state.page = 1;
    await refreshDashboard(true);
    window.alert(
      `DB 초기화가 완료되었습니다. 총 ${Number(payload.deleted_rows || 0).toLocaleString("ko-KR")}개 행을 삭제했습니다.`,
    );
  } catch (error) {
    showError(error.message || "DB 데이터를 초기화하지 못했습니다.");
  } finally {
    elements.databaseReset.disabled = false;
  }
});

elements.pageSize.addEventListener("change", () => {
  state.pageSize = Number(elements.pageSize.value);
  state.page = 1;
  refreshDashboard(true);
});

elements.previousPage.addEventListener("click", () => {
  if (state.page > 1) {
    state.page -= 1;
    refreshDashboard(true);
  }
});

elements.nextPage.addEventListener("click", () => {
  if (state.page < state.totalPages) {
    state.page += 1;
    refreshDashboard(true);
  }
});

elements.refresh.addEventListener("click", () => refreshDashboard(true));

setInterval(() => {
  if (elements.autoRefresh.checked && !document.hidden) {
    refreshDashboard(false);
  }
}, 5000);

refreshDashboard(true);
