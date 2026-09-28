<template>
  <section v-if="match" class="result-section">
    <div class="section-title">
      <span>Результат сканирования</span>
      <h2>{{ match.wine?.name || match.slug }}</h2>
    </div>

    <article class="wine-hero">
      <div class="wine-info">
        <p class="match-label">Лучшее совпадение · {{ formatPercent(match.similarity) }}</p>
        <h3><TermText :text="match.wine?.name" /></h3>
        <a v-if="match.wine?.producer" class="producer-link" :href="match.wine?.source_url || '#'" target="_blank">
          {{ match.wine.producer }}
        </a>
        <div class="rating-pill">
          <Wine :size="18" />
          <span>Народный рейтинг {{ match.wine?.rating || "—" }}</span>
        </div>
        <div class="facts-grid">
          <WineFact label="Регион" :value="regionLine(match.wine)" />
          <WineFact label="Виноград" :value="grapeLine(match.wine)" />
          <WineFact label="Категория" :value="categoryLine(match.wine)" />
          <WineFact label="Оттенок" :value="shadeLine(match.wine)" />
          <WineFact label="Цена" :value="formatPrice(match.wine)" />
        </div>
        <p v-if="match.wine?.description" class="description"><TermText :text="match.wine.description" /></p>
        <div class="hero-actions">
          <button
            class="ui-button"
            :class="isFavorite ? 'secondary' : 'primary'"
            type="button"
            :disabled="favoriteBusy"
            :aria-pressed="isFavorite"
            :title="isFavorite ? 'Убрать из избранного' : ''"
            @click="toggleFavorite"
          >
            <LoaderCircle v-if="favoriteBusy" :size="18" class="spin" />
            <Heart v-else :size="18" :fill="isFavorite ? 'currentColor' : 'none'" />
            {{ isFavorite ? "В избранном" : "В избранное" }}
          </button>
          <a v-if="match.wine?.source_url" class="ui-button secondary as-link" :href="match.wine.source_url" target="_blank" rel="noreferrer">
            Открыть на портале
          </a>
        </div>
      </div>

      <div class="wine-visual">
        <button
          v-if="activePhoto"
          class="photo-stage photo-stage-open"
          type="button"
          :aria-label="`Открыть фото на весь экран: ${match.wine?.name}`"
          @click="openViewer(photoList, Math.max(0, photoList.findIndex((photo) => photo.url === activePhoto)))"
        >
          <img :src="activePhoto" :alt="match.wine?.name" />
          <span class="photo-stage-zoom" aria-hidden="true"><Maximize2 :size="16" /></span>
        </button>
        <div v-else class="photo-stage">
          <span>Фото вина</span>
        </div>
        <div v-if="realPhotos(match.wine).length > 1" class="thumb-strip">
          <button
            v-for="photo in realPhotos(match.wine)"
            :key="photo.id || photo.url"
            type="button"
            :class="{ active: app.selectedPhotos.value[match.wine_id] === photo.url }"
            @click="app.selectedPhotos.value = { ...app.selectedPhotos.value, [match.wine_id]: photo.url }; app.saveSearch()"
          >
            <img :src="photo.url" :alt="match.wine?.name" />
          </button>
        </div>
        <div class="glass-tiles">
          <div>
            <Thermometer :size="20" />
            <span>Подача</span>
            <strong>{{ servingTemperature(match.wine) }}</strong>
          </div>
          <div>
            <Percent :size="20" />
            <span>Крепость</span>
            <strong>{{ alcoholLine(match.wine) }}</strong>
          </div>
          <div class="wide">
            <Utensils :size="20" />
            <span>Сочетание с блюдами</span>
            <strong>{{ foodPairing(match.wine).join(", ") || "—" }}</strong>
          </div>
        </div>
      </div>
    </article>

    <div v-if="generatedPhotos(match.wine).length" class="generated-block">
      <button class="ui-button tertiary" type="button" @click="app.showGenerated.value = !app.showGenerated.value">
        <Images :size="18" />
        {{ app.showGenerated.value ? "Скрыть сгенерированные изображения" : "Показать сгенерированные изображения" }}
      </button>
      <div v-if="app.showGenerated.value" class="generated-grid">
        <button
          v-for="(photo, index) in scenes"
          :key="photo.url"
          class="generated-open"
          type="button"
          :aria-label="`Открыть сцену ${index + 1} из ${scenes.length}`"
          @click="openViewer(scenes, index)"
        >
          <img :src="photo.url" :alt="match.wine?.name" />
        </button>
      </div>
    </div>

    <WineReviews v-if="match.wine" :wine="match.wine" />

    <PhotoViewer v-model:index="viewerIndex" :photos="viewerPhotos" :label="match.wine?.name || 'Фото вина'" />

    <SearchDiagnostics />
  </section>
</template>

<script setup>
import { Heart, Images, LoaderCircle, Maximize2, Percent, Thermometer, Utensils, Wine } from "@lucide/vue";

const props = defineProps({
  match: {
    type: Object,
    default: null,
  },
});

const app = useWineApp();
const {
  alcoholLine,
  categoryLine,
  foodPairing,
  formatPercent,
  formatPrice,
  generatedPhotos,
  grapeLine,
  mainPhoto,
  realPhotos,
  regionLine,
  servingTemperature,
  shadeLine,
} = useWineFormat();

const favoriteBusy = ref(false);

// просмотр на весь экран: настоящие фото — с того, что сейчас выбрано; сцены — отдельным списком
const viewerPhotos = ref([]);
const viewerIndex = ref(null);
const photoList = computed(() => {
  const photos = realPhotos(props.match?.wine).map((photo) => ({ url: photo.url }));
  return photos.length ? photos : activePhoto.value ? [{ url: activePhoto.value }] : [];
});
const scenes = computed(() =>
  generatedPhotos(props.match?.wine).map((photo) => ({ url: photo.url, caption: "Сцена сгенерирована по фото бутылки" })),
);

function openViewer(list, index) {
  if (!list.length) return;
  viewerPhotos.value = list;
  viewerIndex.value = index;
}
const isFavorite = computed(() =>
  app.favorites.value.items.some((item) => item.wine?.id === props.match?.wine_id),
);

// кнопка-переключатель: видно, что вино добавилось, повторное нажатие убирает
async function toggleFavorite() {
  favoriteBusy.value = true;
  try {
    if (isFavorite.value) await app.removeFavorite(props.match.wine_id);
    else await app.addFavorite(props.match);
  } finally {
    favoriteBusy.value = false;
  }
}

const activePhoto = computed(() => {
  if (!props.match?.wine) return "";
  return app.selectedPhotos.value[props.match.wine_id] || mainPhoto(props.match.wine);
});
</script>
