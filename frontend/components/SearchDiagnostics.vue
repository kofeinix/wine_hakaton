<template>
  <section class="diagnostics">
    <button class="ui-button secondary" type="button" @click="app.showDiagnostics.value = !app.showDiagnostics.value">
      <Activity :size="18" />
      {{ app.showDiagnostics.value ? "Скрыть диагностику" : "Показать диагностику" }}
    </button>

    <div v-if="app.showDiagnostics.value && response" class="diag">
      <div class="diag-stats">
        <div v-for="stat in stats" :key="stat.label" class="diag-stat">
          <span>{{ stat.label }}</span>
          <strong>{{ stat.value }}</strong>
          <small>{{ stat.note }}</small>
        </div>
      </div>

      <div class="diag-row">
        <article class="diag-card diag-photo-card">
          <h4>Фото и детекция</h4>
          <div v-if="imageUrl && original" class="diag-photo">
            <img :src="imageUrl" alt="Фото, по которому шёл поиск" />
            <svg :viewBox="`0 0 ${original.width} ${original.height}`" preserveAspectRatio="none" aria-hidden="true">
              <rect
                v-for="box in boxes"
                :key="box.key"
                :class="['bbox', box.key]"
                :x="box.x"
                :y="box.y"
                :width="box.w"
                :height="box.h"
                vector-effect="non-scaling-stroke"
              />
            </svg>
            <span
              v-for="box in boxes"
              :key="`tag-${box.key}`"
              :class="['bbox-tag', box.key, { inside: box.y / original.height < 0.08 }]"
              :style="{ left: `${(box.x / original.width) * 100}%`, top: `${(box.y / original.height) * 100}%` }"
            >
              {{ box.label }} · {{ formatPercent(box.confidence) }}
            </span>
          </div>
          <p v-else class="diag-muted">Фото недоступно — диагностика пришла без исходного изображения.</p>

          <div class="diag-crops">
            <div v-for="crop in cropCards" :key="crop.key" class="diag-crop">
              <div
                v-if="crop.style"
                class="diag-crop-image"
                :style="crop.style"
                role="img"
                :aria-label="`Кроп: ${crop.label}`"
              />
              <div v-else class="diag-crop-image empty">нет</div>
              <div>
                <strong><i :class="['swatch', crop.key]" />{{ crop.label }}</strong>
                <span>{{ crop.note }}</span>
              </div>
            </div>
          </div>
        </article>

        <article class="diag-card">
          <h4>Время этапов</h4>
          <p class="diag-muted">Всего {{ ms(timings.total_ms) }}. OCR идёт параллельно с эмбеддингами и Qdrant.</p>
          <div class="timing-list">
            <div v-for="stage in timingRows" :key="stage.key" class="timing-row" :title="`${stage.label}: ${ms(stage.value)}`">
              <span>{{ stage.label }}</span>
              <div class="timing-track">
                <div class="timing-bar" :class="{ parallel: stage.parallel }" :style="{ width: `${stage.share}%` }" />
              </div>
              <strong>{{ ms(stage.value) }}</strong>
            </div>
          </div>

          <h4 class="diag-subtitle">Визуальный поиск</h4>
          <dl class="diag-params">
            <div v-for="param in visualParams" :key="param.label">
              <dt>{{ param.label }}</dt>
              <dd>{{ param.value }}</dd>
            </div>
          </dl>
          <div v-if="viewWeights.length" class="weight-list">
            <div v-for="view in viewWeights" :key="view.key" class="weight-row" :title="`${view.label}: вес ${view.weight}`">
              <span>{{ view.label }}</span>
              <div class="timing-track">
                <div class="timing-bar" :class="{ inactive: !view.active }" :style="{ width: `${view.weight * 100}%` }" />
              </div>
              <strong>{{ view.weight }}</strong>
            </div>
          </div>
        </article>
      </div>

      <article class="diag-card">
        <div class="diag-card-head">
          <h4>Кандидаты и скоры</h4>
          <div class="diag-legend">
            <span><i class="swatch visual" />визуальный скор</span>
            <span><i class="swatch ocr" />OCR-бонус</span>
            <span><i class="swatch penalty" />OCR-штраф</span>
          </div>
        </div>
        <p class="diag-muted">
          {{ formula }} Нажмите на строку, чтобы увидеть признаки OCR.
        </p>
        <div class="candidate-list">
          <div v-for="(row, index) in candidates" :key="row.wineId" class="candidate">
            <button
              class="candidate-row"
              type="button"
              :aria-expanded="expanded === row.wineId"
              @click="expanded = expanded === row.wineId ? null : row.wineId"
            >
              <span class="candidate-rank">{{ index + 1 }}</span>
              <span class="candidate-name">{{ row.name }}</span>
              <span class="score-track" :title="`визуальный ${fmt(row.visual)} ${row.ocr >= 0 ? '+' : '−'} OCR ${fmt(Math.abs(row.ocr))} = ${fmt(row.final)}`">
                <span class="score-bar visual" :style="{ width: `${pct(Math.min(row.visual, row.final))}%` }" />
                <span
                  v-if="row.ocr"
                  class="score-bar"
                  :class="row.ocr > 0 ? 'ocr' : 'penalty'"
                  :style="{ width: `${pct(Math.abs(row.ocr))}%` }"
                />
              </span>
              <span class="candidate-score">
                <strong>{{ fmt(row.final) }}</strong>
                <small>{{ fmt(row.visual) }} {{ row.ocr >= 0 ? "+" : "−" }} {{ fmt(Math.abs(row.ocr)) }}</small>
              </span>
              <ChevronDown :size="18" class="candidate-chevron" />
            </button>
            <div v-if="expanded === row.wineId" class="candidate-features">
              <p v-if="!row.features" class="diag-muted">Для этого кандидата OCR-признаков нет (OCR пропущен или не распознал текст).</p>
              <dl v-else class="diag-params features">
                <div v-for="feature in featureRows(row.features)" :key="feature.key" :class="feature.tone">
                  <dt>{{ feature.label }}</dt>
                  <dd>{{ feature.value }}</dd>
                </div>
              </dl>
            </div>
          </div>
        </div>
      </article>

      <div class="diag-row">
        <article class="diag-card">
          <h4>Распознанный текст</h4>
          <p class="diag-muted">{{ ocrNote }}</p>
          <pre v-if="response.ocr?.text" class="diag-text">{{ response.ocr.text }}</pre>
          <p v-else class="diag-muted">Текст не распознан.</p>
          <template v-if="rerank.normalized_text">
            <h5>Нормализованный</h5>
            <pre class="diag-text small">{{ rerank.normalized_text }}</pre>
          </template>
        </article>

        <article class="diag-card">
          <h4>Сырой ответ</h4>
          <details class="diag-raw">
            <summary>Показать JSON диагностики</summary>
            <pre class="diag-text small">{{ rawJson }}</pre>
          </details>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup>
import { Activity, ChevronDown } from "@lucide/vue";

const app = useWineApp();
const { formatPercent } = useWineFormat();
const expanded = ref(null);

const response = computed(() => app.searchResponse.value);
const imageUrl = computed(() => app.searchedImageUrl.value);
const diagnostics = computed(() => response.value?.diagnostics || {});
const rerank = computed(() => diagnostics.value.ocr_rerank || {});
const visual = computed(() => diagnostics.value.visual || {});
const timings = computed(() => response.value?.timings_ms || {});
const crops = computed(() => response.value?.crops || {});
const original = computed(() => response.value?.image || (crops.value.original?.available ? crops.value.original : null));
const rawJson = computed(() => JSON.stringify(diagnostics.value, null, 2));
const formula = computed(() => (rerank.value.formula ? `${rerank.value.formula.split(" (")[0]}.` : "final = visual + OCR."));

const VIEW_LABELS = { original: "Всё фото", bottle_crop: "Бутылка", label_crop: "Этикетка" };

function ms(value) {
  if (typeof value !== "number") return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(1)} с` : `${Math.round(value)} мс`;
}

function fmt(value) {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

// координаты этикетки могут быть внутри кропа бутылки — переводим в систему исходного фото
function absoluteBox(key) {
  const crop = crops.value[key];
  if (!crop?.available || !crop.box || crop.source_view === "original_fallback") return null;
  let [x1, y1, x2, y2] = crop.box;
  if (crop.source_view === "bottle_crop") {
    const bottle = crops.value.bottle_crop?.box;
    if (!bottle) return null;
    x1 += bottle[0];
    x2 += bottle[0];
    y1 += bottle[1];
    y2 += bottle[1];
  }
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1, confidence: crop.confidence };
}

const boxes = computed(() =>
  ["bottle_crop", "label_crop"]
    .map((key) => {
      const box = absoluteBox(key);
      return box && { ...box, key, label: VIEW_LABELS[key] };
    })
    .filter(Boolean),
);

function cropStyle(box) {
  const W = original.value?.width;
  const H = original.value?.height;
  if (!imageUrl.value || !box || !W || !H || box.w <= 0 || box.h <= 0) return null;
  // фрагмент исходного фото через background: без canvas и повторной загрузки
  return {
    aspectRatio: `${box.w} / ${box.h}`,
    backgroundImage: `url(${imageUrl.value})`,
    backgroundSize: `${(W / box.w) * 100}% ${(H / box.h) * 100}%`,
    backgroundPosition: `${W === box.w ? 0 : (box.x / (W - box.w)) * 100}% ${H === box.h ? 0 : (box.y / (H - box.h)) * 100}%`,
  };
}

const cropCards = computed(() =>
  ["bottle_crop", "label_crop"].map((key) => {
    const crop = crops.value[key] || {};
    const box = absoluteBox(key);
    let note = "не найдено";
    if (crop.source_view === "original_fallback") note = "не найдена — используется всё фото";
    else if (crop.available) {
      const source = crop.source_view === "bottle_crop" ? " · найдена внутри бутылки" : "";
      note = `${crop.width}×${crop.height} px · уверенность ${formatPercent(crop.confidence)}${source}`;
    }
    return { key, label: VIEW_LABELS[key], note, style: cropStyle(box) };
  }),
);

const top = computed(() => response.value?.results?.[0]);

const searchMode = computed(() => {
  const search = response.value?.search || {};
  if (search.label_photo_mode) return "режим: крупная этикетка";
  if (search.multi_wine_mode) return `режим: полка (${search.labels_detected} этикеток) — без всего кадра`;
  return "режим: вся бутылка";
});

const stats = computed(() => {
  const ocr = response.value?.ocr || {};
  const gap = rerank.value.visual_gap;
  const threshold = rerank.value.ocr_skip_visual_gap;
  let ocrValue = "не применён";
  if (ocr.reason === "timeout_budget") ocrValue = "не успел";
  else if (ocr.skipped) ocrValue = "пропущен";
  else if (ocr.applied) ocrValue = ocr.cached ? "из кеша" : "применён";
  return [
    {
      label: "Статус",
      value: response.value?.status === "found" ? "найдено" : "не найдено",
      note: top.value ? `лучший скор ${fmt(top.value.final_score)}` : "кандидатов нет",
    },
    {
      label: "Отрыв визуального top-1",
      value: typeof gap === "number" ? fmt(gap) : "—",
      note: typeof threshold === "number" ? `порог пропуска OCR ${fmt(threshold)}` : "",
    },
    {
      label: "OCR",
      value: ocrValue,
      note: ocr.source_view ? `читали: ${VIEW_LABELS[ocr.source_view] || ocr.source_view}` : "",
    },
    {
      label: "Кандидатов",
      value: response.value?.search?.candidates ?? "—",
      note: searchMode.value,
    },
    { label: "Время", value: ms(timings.value.total_ms), note: `OCR ${ms(timings.value.ocr_ms)}` },
  ];
});

const timingRows = computed(() => {
  const t = timings.value;
  const total = t.total_ms || 1;
  return [
    { key: "crops_ms", label: "YOLO-кропы" },
    { key: "embedding_ms", label: "Эмбеддинги" },
    { key: "vector_search_ms", label: "Qdrant" },
    { key: "ocr_ms", label: "OCR (параллельно)", parallel: true },
    { key: "rerank_ms", label: "OCR-реранк" },
  ]
    .filter((row) => typeof t[row.key] === "number")
    .map((row) => ({ ...row, value: t[row.key], share: Math.min(100, (t[row.key] / total) * 100) }));
});

const visualParams = computed(() => {
  const v = visual.value;
  return [
    { label: "Ракурсы", value: (v.active_views || response.value?.search?.active_views || []).map((key) => VIEW_LABELS[key] || key).join(", ") || "—" },
    { label: "Доля этикетки в кадре", value: typeof v.label_area_ratio === "number" ? formatPercent(v.label_area_ratio) : "—" },
    { label: "Порог «крупной этикетки»", value: typeof v.label_photo_area_threshold === "number" ? formatPercent(v.label_photo_area_threshold) : "—" },
    { label: "Top-k на ракурс", value: v.per_view_top_k ?? "—" },
    { label: "Энкодер", value: v.collection_encoder || "—" },
    { label: "Агрегация", value: v.aggregation || "—" },
  ];
});

const viewWeights = computed(() => {
  const weights = visual.value.weights || {};
  const active = new Set(visual.value.active_views || []);
  return Object.entries(weights).map(([key, weight]) => ({ key, label: VIEW_LABELS[key] || key, weight, active: active.has(key) }));
});

const candidates = computed(() => {
  const results = response.value?.results || [];
  const names = Object.fromEntries(results.map((item) => [item.wine_id, item.wine?.name || item.slug]));
  const scored = rerank.value.candidate_scores || [];
  const features = Object.fromEntries(scored.map((item) => [item.wine_id, item]));
  const rows = scored.length
    ? scored.map((item) => ({
        wineId: item.wine_id,
        visual: item.visual_score,
        ocr: item.ocr_bonus ?? 0,
        final: item.final_score,
      }))
    : results.map((item) => ({ wineId: item.wine_id, visual: item.visual_score, ocr: item.ocr_score, final: item.final_score }));
  return rows.map((row) => ({
    ...row,
    name: names[row.wineId] || `${row.wineId.slice(0, 8)}… (вне выдачи)`,
    features: features[row.wineId] || null,
  }));
});

const scoreMax = computed(() =>
  Math.max(0.01, ...candidates.value.map((row) => Math.max(row.visual, row.final, row.visual + Math.max(row.ocr, 0)))),
);

function pct(value) {
  return (value / scoreMax.value) * 100;
}

const FEATURES = [
  ["name_coverage", "Название: покрытие", "share"],
  ["name_window", "Название: окно", "share"],
  ["producer_coverage", "Производитель: покрытие", "share"],
  ["producer_window", "Производитель: окно", "share"],
  ["producer_detected", "Производитель найден", "flag"],
  ["producer_contradiction", "Другой производитель", "bad"],
  ["grape_coverage_max", "Сорт: покрытие", "share"],
  ["grape_hit", "Сорт совпал", "flag"],
  ["grape_contradiction", "Другой сорт", "bad"],
  ["color_own", "Цвет: свой", "share"],
  ["color_contradiction", "Цвет противоречит", "bad"],
  ["sugar_own", "Сахар: свой", "share"],
  ["sugar_contradiction", "Сахар противоречит", "bad"],
  ["alcohol_match", "Крепость совпала", "flag"],
  ["alcohol_mismatch", "Крепость не совпала", "bad"],
  ["unique_evidence", "Уникальные слова (IDF)", "num"],
  ["unique_evidence_share", "Доля уникальных слов", "share"],
  ["text_tokens", "Токенов в тексте", "int"],
];

function featureRows(features) {
  return FEATURES.filter(([key]) => typeof features[key] === "number").map(([key, label, kind]) => {
    const value = features[key];
    let text = value.toFixed(2);
    let tone = "";
    if (kind === "share") text = formatPercent(value);
    if (kind === "int") text = String(Math.round(value));
    if (kind === "flag") {
      text = value ? "да" : "нет";
      tone = value ? "good" : "";
    }
    if (kind === "bad") {
      text = value ? "да" : "нет";
      tone = value ? "bad" : "";
    }
    return { key, label, value: text, tone };
  });
}

const ocrNote = computed(() => {
  const ocr = response.value?.ocr || {};
  const reasons = {
    timeout_budget: "OCR не уложился во время: распознавание текста не успело, ответ — только по изображению.",
    skipped_confident_visual: "OCR не понадобился: визуальный результат достаточно уверенный.",
    llm_unavailable: "OCR недоступен: модель распознавания текста не подключена, ответ — только по изображению.",
    ocr_failed: "OCR завершился ошибкой: ответ — только по изображению.",
    empty_ocr_text: "OCR не нашёл текста на этикетке.",
  };
  if (reasons[ocr.reason]) return reasons[ocr.reason];
  if (ocr.skipped) return reasons.skipped_confident_visual;
  const source = VIEW_LABELS[ocr.source_view] || ocr.source_view || "—";
  const status = rerank.value.ocr_status ? ` · статус ${rerank.value.ocr_status}` : "";
  return `Источник: ${source}${ocr.cached ? " · из кеша" : ""}${status}`;
});
</script>
