<template>
  <section class="sommelier-page">
    <div class="page-heading">
      <p class="breadcrumbs">Главная › Сомелье</p>
      <h1>Подберём вино под ваш случай</h1>
      <p>
        Отметьте, для чего вино, что будет на столе и каким оно должно быть. Подборка строится по данным
        карточек каталога, а у каждого вина написано, чем именно оно подошло.
      </p>
    </div>

    <label class="sm-search">
      <Search :size="20" />
      <input v-model="filters.q" type="search" placeholder="Название, производитель или сорт" autocomplete="off" />
      <button v-if="filters.q" type="button" aria-label="Очистить поиск" @click="filters.q = ''">
        <X :size="18" />
      </button>
    </label>

    <div class="sm-quick">
      <div v-for="group in quickGroups" :key="group.key" class="sm-quick-group">
        <h2>{{ group.title }}</h2>
        <div class="sm-chips">
          <button
            v-for="option in facets?.[group.facet] || []"
            :key="option.value"
            type="button"
            class="sm-chip"
            :class="{ selected: isSelected(group.key, option.value) }"
            :aria-pressed="isSelected(group.key, option.value)"
            :disabled="!option.count && !isSelected(group.key, option.value)"
            @click="toggle(group.key, option.value)"
          >
            {{ option.label }}
            <span>{{ option.count }}</span>
          </button>
        </div>
      </div>
    </div>

    <div class="sm-layout">
      <SommelierFilters
        :facets="facets"
        :filters="filters"
        :open="filtersOpen"
        :total="total"
        @toggle="toggle"
        @rating="filters.min_rating = $event"
        @close="filtersOpen = false"
        @reset="resetFilters"
      />

      <div class="sm-results">
        <div class="sm-results-head">
          <div>
            <h2>{{ hasCriteria ? "Под ваш запрос" : "Выбор сомелье" }}</h2>
            <span>{{ total }} {{ winesWord(total) }}</span>
          </div>
          <div class="sm-results-actions">
            <button class="sm-chip sm-filter-button" type="button" @click="filtersOpen = true">
              <SlidersHorizontal :size="16" />
              Фильтр
              <span v-if="filterCount">{{ filterCount }}</span>
            </button>
            <select v-model="filters.sort" aria-label="Сортировка">
              <option value="relevance">Сначала подходящие</option>
              <option value="rating">По рейтингу</option>
              <option value="name">По названию</option>
            </select>
            <button v-if="hasCriteria" class="ui-button transparent small" type="button" @click="resetAll">
              Сбросить всё
            </button>
          </div>
        </div>

        <p v-if="correctedQuery" class="sm-notice">
          По запросу «{{ filters.q }}» ничего нет — показаны результаты для «{{ correctedQuery }}».
        </p>
        <p v-if="error" class="error-message">{{ error }}</p>

        <div v-if="items.length" class="wine-grid sm-grid">
          <button
            v-for="item in items"
            :key="item.wine.id"
            class="wine-card sm-card"
            type="button"
            @click="app.openWine(item.wine, { context: 'Сомелье' })"
          >
            <span v-if="item.wine.rating" class="rating-badge">🍷 {{ formatRating(item.wine.rating) }}</span>
            <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" :alt="item.wine.name" loading="lazy" />
            <span v-else class="no-photo">Фото</span>
            <strong>{{ item.wine.name }}</strong>
            <small>{{ item.wine.producer }}</small>
            <span class="sm-card-meta">{{ metaLine(item.wine) }}</span>
            <ul v-if="item.reasons.length" class="sm-reasons">
              <li v-for="reason in item.reasons.slice(0, 3)" :key="reason">
                <Check :size="14" />
                {{ reason }}
              </li>
            </ul>
          </button>
        </div>

        <div v-else-if="!loading" class="sm-empty">
          <Wine :size="40" />
          <p>Под такие условия вин нет. Уберите часть фильтров или поменяйте запрос.</p>
          <button class="ui-button secondary" type="button" @click="resetAll">Сбросить всё</button>
        </div>

        <div v-if="loading && !items.length" class="sm-loading">
          <LoaderCircle :size="28" class="spin" />
        </div>

        <button
          v-if="items.length < total"
          class="ui-button primary large sm-more"
          type="button"
          :disabled="loading"
          @click="loadMore"
        >
          <LoaderCircle v-if="loading" :size="18" class="spin" />
          Показать ещё
        </button>
      </div>
    </div>

    <SommelierRegions />
  </section>
</template>

<script setup>
import { Check, LoaderCircle, Search, SlidersHorizontal, Wine, X } from "@lucide/vue";

const PAGE = 12;
const LIST_KEYS = ["dish", "taste", "type", "color", "sugar", "region", "grape"];

const app = useWineApp();
const { mainPhoto } = useWineFormat();

const quickGroups = [
  { key: "occasion", facet: "occasions", title: "Для чего" },
  { key: "dish", facet: "dishes", title: "Что на столе" },
  { key: "taste", facet: "tastes", title: "Каким должно быть" },
];

const filters = reactive(emptyFilters());
const items = ref([]);
const total = ref(0);
const facets = ref(null);
const correctedQuery = ref(null);
const loading = ref(false);
const error = ref("");
const filtersOpen = ref(false);
let requestId = 0;
let debounce = null;

// фильтры панели (без поиска, повода, блюд и вкуса) — для счётчика на мобильной кнопке
const filterCount = computed(
  () =>
    ["type", "color", "sugar", "region", "grape"].reduce((sum, key) => sum + filters[key].length, 0) +
    (filters.min_rating ? 1 : 0),
);
const hasCriteria = computed(
  () => Boolean(filters.q.trim() || filters.occasion || filters.dish.length || filters.taste.length) || filterCount.value > 0,
);

function emptyFilters() {
  return {
    q: "",
    occasion: null,
    dish: [],
    taste: [],
    type: [],
    color: [],
    sugar: [],
    region: [],
    grape: [],
    min_rating: null,
    sort: "relevance",
  };
}

function isSelected(key, value) {
  return key === "occasion" ? filters.occasion === value : filters[key].includes(value);
}

function toggle(key, value) {
  if (key === "occasion") {
    filters.occasion = filters.occasion === value ? null : value;
    return;
  }
  const list = filters[key];
  const index = list.indexOf(value);
  if (index >= 0) list.splice(index, 1);
  else list.push(value);
}

function resetFilters() {
  for (const key of ["type", "color", "sugar", "region", "grape"]) filters[key] = [];
  filters.min_rating = null;
}

function resetAll() {
  Object.assign(filters, emptyFilters());
}

function params(offset) {
  const query = new URLSearchParams({ limit: String(PAGE), offset: String(offset), sort: filters.sort });
  if (filters.q.trim()) query.set("q", filters.q.trim());
  if (filters.occasion) query.set("occasion", filters.occasion);
  if (filters.min_rating) query.set("min_rating", String(filters.min_rating));
  for (const key of LIST_KEYS) for (const value of filters[key]) query.append(key, value);
  return query;
}

async function load({ append = false } = {}) {
  const id = ++requestId;
  loading.value = true;
  error.value = "";
  try {
    const response = await $fetch(`/api/v1/sommelier/search?${params(append ? items.value.length : 0)}`);
    if (id !== requestId) return; // пришёл ответ на устаревший запрос
    items.value = append ? [...items.value, ...response.items] : response.items;
    total.value = response.total;
    facets.value = response.facets;
    correctedQuery.value = response.corrected_query;
  } catch {
    if (id === requestId) error.value = "Не удалось загрузить подборку. Попробуйте ещё раз.";
  } finally {
    if (id === requestId) loading.value = false;
  }
}

function loadMore() {
  load({ append: true });
}

// текст — с задержкой, чтобы не искать на каждую букву; остальное — сразу
watch(
  () => filters.q,
  () => {
    clearTimeout(debounce);
    debounce = setTimeout(load, 300);
  },
);
watch(
  () => [filters.occasion, filters.min_rating, filters.sort, ...LIST_KEYS.map((key) => filters[key].join("|"))].join("§"),
  () => {
    clearTimeout(debounce);
    load();
  },
);

onMounted(load);
onBeforeUnmount(() => clearTimeout(debounce));

function formatRating(value) {
  return Number(value).toFixed(2);
}

function metaLine(wine) {
  return [wine.color, wine.sugar, wine.region].filter(Boolean).join(" · ");
}

function winesWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "вино";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "вина";
  return "вин";
}
</script>
