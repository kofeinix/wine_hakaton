<template>
  <section v-if="source" class="bottle-picker" :class="{ compact: !multi && !drawing && !app.pick.value }">
    <div class="bottle-picker-head">
      <ScanSearch :size="22" />
      <div>
        <h3 v-if="multi">На фото несколько вин · {{ labels.length }} {{ plural(labels.length) }}</h3>
        <h3 v-else>Нашли не ту бутылку?</h3>
        <p v-if="drawing">Проведите по фото, чтобы обвести нужную бутылку или этикетку.</p>
        <p v-else-if="multi">
          Выделена бутылка, по которой мы искали. Если нужна другая — нажмите на её этикетку или обведите вручную.
          Точнее всего поиск работает, когда в кадре одна бутылка крупным планом.
        </p>
        <p v-else>Обведите нужную бутылку на фото вручную — поищем только по ней.</p>
      </div>
    </div>

    <div v-if="multi || drawing || app.pick.value" ref="scroller" class="bottle-picker-scroll">
      <div
        ref="stage"
        class="bottle-picker-photo"
        :class="{ busy: app.isSearching.value, drawing }"
        @pointerdown="startDraw"
        @pointermove="moveDraw"
        @pointerup="endDraw"
        @pointercancel="cancelDraw"
      >
        <img :src="source.url" alt="Исходное фото" draggable="false" />
        <button
          v-for="(label, index) in labels"
          :key="label.box.join(',')"
          type="button"
          class="label-hit"
          :class="{ selected: index === selectedIndex }"
          :style="boxStyle(label.box)"
          :aria-label="`Этикетка ${index + 1}, уверенность ${formatPercent(label.confidence)}`"
          :aria-pressed="index === selectedIndex"
          :disabled="app.isSearching.value || drawing"
          @click="chooseLabel(index)"
        />
        <div v-if="app.pick.value?.kind === 'manual'" class="manual-box" :style="boxStyle(app.pick.value.box)" />
        <div v-if="draft" class="manual-box draft" :style="boxStyle(draft)" />
        <div v-if="app.isSearching.value" class="bottle-picker-busy">
          <LoaderCircle :size="28" class="spin" />
          Ищем выбранное вино
        </div>
      </div>
    </div>

    <div class="bottle-picker-actions">
      <button
        class="ui-button small"
        :class="drawing ? 'secondary' : 'tertiary'"
        type="button"
        :disabled="app.isSearching.value"
        @click="drawing = !drawing"
      >
        <Crop v-if="!drawing" :size="16" />
        <X v-else :size="16" />
        {{ drawing ? "Отменить выделение" : "Выделить вручную" }}
      </button>
      <button
        v-if="app.pick.value"
        class="ui-button tertiary small"
        type="button"
        :disabled="app.isSearching.value"
        @click="auto"
      >
        <Sparkles :size="16" />
        Автовыбор
      </button>
    </div>
  </section>
</template>

<script setup>
import { Crop, LoaderCircle, ScanSearch, Sparkles, X } from "@lucide/vue";

const LABEL_MARGIN = 0.1; // запас вокруг этикетки: соседние почти не попадают в кадр
const MIN_MANUAL_SIDE = 24; // px исходного фото — меньше считаем случайным касанием

const app = useWineApp();
const { formatPercent } = useWineFormat();
const drawing = ref(false);
const draft = ref(null);
const stage = ref(null);
const scroller = ref(null);
let drawStart = null;

const source = computed(() => app.photoSource.value);
const labels = computed(() => source.value?.labels || []);
const multi = computed(() => labels.value.length > 1);

function plural(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "этикетка";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "этикетки";
  return "этикеток";
}

function boxStyle([x1, y1, x2, y2]) {
  const { width, height } = source.value;
  return {
    left: `${(x1 / width) * 100}%`,
    top: `${(y1 / height) * 100}%`,
    width: `${((x2 - x1) / width) * 100}%`,
    height: `${((y2 - y1) / height) * 100}%`,
  };
}

function clampBox([x1, y1, x2, y2]) {
  const { width, height } = source.value;
  return [Math.max(0, x1), Math.max(0, y1), Math.min(width, x2), Math.min(height, y2)].map(Math.round);
}

// подсветка: выбор пользователя; при автовыборе — этикетка, которую взял бэкенд
const selectedIndex = computed(() => {
  const pick = app.pick.value;
  if (pick) return pick.kind === "label" ? pick.index : -1;
  const response = app.searchResponse.value;
  const crop = response?.crops?.label_crop;
  if (!crop?.available || !crop.box || crop.source_view === "original_fallback") return -1;
  let [x1, y1, x2, y2] = crop.box;
  if (crop.source_view === "bottle_crop") {
    const bottle = response.crops.bottle_crop?.box;
    if (!bottle) return -1;
    [x1, y1, x2, y2] = [x1 + bottle[0], y1 + bottle[1], x2 + bottle[0], y2 + bottle[1]];
  }
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  return labels.value.findIndex(({ box }) => cx >= box[0] && cx <= box[2] && cy >= box[1] && cy <= box[3]);
});

function chooseLabel(index) {
  if (index === selectedIndex.value) return;
  const [x1, y1, x2, y2] = labels.value[index].box;
  const dx = (x2 - x1) * LABEL_MARGIN;
  const dy = (y2 - y1) * LABEL_MARGIN;
  app.searchArea(clampBox([x1 - dx, y1 - dy, x2 + dx, y2 + dy]), "label", { index });
}

function auto() {
  drawing.value = false;
  app.searchWine();
}

// --- ручная рамка ---
function pointToImage(event) {
  const rect = stage.value.getBoundingClientRect();
  const scale = source.value.width / rect.width;
  return [
    Math.min(source.value.width, Math.max(0, (event.clientX - rect.left) * scale)),
    Math.min(source.value.height, Math.max(0, (event.clientY - rect.top) * scale)),
  ];
}

function startDraw(event) {
  if (!drawing.value || app.isSearching.value) return;
  event.preventDefault();
  try {
    stage.value.setPointerCapture(event.pointerId); // рамку можно дотянуть и за пределы фото
  } catch {
    // без захвата тоже работает, просто в пределах фото
  }
  drawStart = pointToImage(event);
  draft.value = [...drawStart, ...drawStart];
}

function moveDraw(event) {
  if (!drawStart) return;
  const [x, y] = pointToImage(event);
  draft.value = [Math.min(drawStart[0], x), Math.min(drawStart[1], y), Math.max(drawStart[0], x), Math.max(drawStart[1], y)];
}

function endDraw() {
  if (!drawStart) return;
  const box = draft.value;
  drawStart = null;
  draft.value = null;
  if (!box || box[2] - box[0] < MIN_MANUAL_SIDE || box[3] - box[1] < MIN_MANUAL_SIDE) return;
  drawing.value = false;
  app.searchArea(clampBox(box), "manual");
}

function cancelDraw() {
  drawStart = null;
  draft.value = null;
}

function onKeydown(event) {
  if (event.key === "Escape" && drawing.value) {
    cancelDraw();
    drawing.value = false;
  }
}

onMounted(() => document.addEventListener("keydown", onKeydown));
onBeforeUnmount(() => document.removeEventListener("keydown", onKeydown));

// на телефоне фото шире экрана: прокручиваем к выбранной этикетке
watch(
  () => [selectedIndex.value, source.value?.url],
  async () => {
    await nextTick();
    const el = scroller.value?.querySelector(".label-hit.selected, .manual-box");
    if (!el || scroller.value.scrollWidth <= scroller.value.clientWidth) return;
    scroller.value.scrollLeft = el.offsetLeft + el.offsetWidth / 2 - scroller.value.clientWidth / 2;
  },
  { immediate: true },
);
</script>
