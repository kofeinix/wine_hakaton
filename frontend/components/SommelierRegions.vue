<template>
  <section class="sm-regions" aria-labelledby="sm-regions-title">
    <div class="sm-regions-head">
      <h2 id="sm-regions-title">Вина по регионам</h2>
      <p>Весь каталог по алфавиту — удобно, если знаете, откуда вино.</p>
    </div>

    <div class="sm-chips sm-region-tabs" role="tablist" aria-label="Регионы">
      <button
        v-for="option in regions"
        :key="option.value"
        type="button"
        role="tab"
        class="sm-chip"
        :class="{ selected: option.value === region }"
        :aria-selected="option.value === region"
        @click="region = option.value"
      >
        {{ option.label }}
        <span>{{ option.count }}</span>
      </button>
    </div>

    <label class="sm-filter-search sm-region-search">
      <input v-model="filter" type="search" :placeholder="`Поиск по винам региона ${region || ''}`" />
      <Search :size="18" />
    </label>

    <ol class="sm-region-list">
      <li v-for="item in items" :key="item.wine.id">
        <button type="button" @click="app.openWine(item.wine, { context: `Регион: ${region}` })">
          <span class="sm-region-photo">
            <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" alt="" loading="lazy" />
          </span>
          <span class="sm-region-body">
            <strong>{{ item.wine.name }}</strong>
            <small>{{ [item.wine.producer, item.wine.color, item.wine.sugar].filter(Boolean).join(" · ") }}</small>
          </span>
          <span v-if="item.wine.rating" class="sm-region-rating">🍷 {{ Number(item.wine.rating).toFixed(2) }}</span>
        </button>
      </li>
    </ol>

    <p v-if="!loading && !items.length" class="profile-empty">Ничего не найдено.</p>
    <p v-if="error" class="error-message">{{ error }}</p>

    <button v-if="items.length < total" class="ui-button secondary sm-more" type="button" :disabled="loading" @click="load(true)">
      <LoaderCircle v-if="loading" :size="18" class="spin" />
      Ещё {{ Math.min(PAGE, total - items.length) }} из {{ total - items.length }}
    </button>
  </section>
</template>

<script setup>
import { LoaderCircle, Search } from "@lucide/vue";

const PAGE = 20;

const app = useWineApp();
const { mainPhoto } = useWineFormat();
const regions = ref([]);
const region = ref("");
const filter = ref("");
const items = ref([]);
const total = ref(0);
const loading = ref(false);
const error = ref("");
let requestId = 0;
let debounce = null;

async function load(append = false) {
  const id = ++requestId;
  loading.value = true;
  error.value = "";
  const query = new URLSearchParams({ sort: "name", limit: String(PAGE), offset: String(append ? items.value.length : 0) });
  if (region.value) query.set("region", region.value);
  if (filter.value.trim()) query.set("q", filter.value.trim());
  try {
    const response = await $fetch(`/api/v1/sommelier/search?${query}`);
    if (id !== requestId) return;
    items.value = append ? [...items.value, ...response.items] : response.items;
    total.value = response.total;
    // счётчики регионов не зависят от выбранного региона (фильтр по своему измерению не учитывается)
    if (!filter.value.trim()) regions.value = response.facets.regions.filter((option) => option.count);
    if (!region.value && regions.value.length) region.value = regions.value[0].value;
  } catch {
    if (id === requestId) error.value = "Не удалось загрузить список";
  } finally {
    if (id === requestId) loading.value = false;
  }
}

watch(region, () => load());
watch(filter, () => {
  clearTimeout(debounce);
  debounce = setTimeout(() => load(), 300);
});

onMounted(() => load());
onBeforeUnmount(() => clearTimeout(debounce));
</script>
