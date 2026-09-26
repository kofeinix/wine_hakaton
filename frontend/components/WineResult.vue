<template>
  <section v-if="match" class="result-section">
    <div class="section-title">
      <span>Результат сканирования</span>
      <h2>{{ match.wine?.name || match.slug }}</h2>
    </div>

    <article class="wine-hero">
      <div class="wine-info">
        <p class="match-label">Лучшее совпадение · {{ formatPercent(match.final_score) }}</p>
        <h3>{{ match.wine?.name }}</h3>
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
        <p v-if="match.wine?.description" class="description">{{ match.wine.description }}</p>
        <div class="hero-actions">
          <button class="ui-button primary" type="button" @click="app.addFavorite(match)">
            <Heart :size="18" />
            В избранное
          </button>
          <a v-if="match.wine?.source_url" class="ui-button secondary as-link" :href="match.wine.source_url" target="_blank" rel="noreferrer">
            Открыть на портале
          </a>
        </div>
      </div>

      <div class="wine-visual">
        <div class="photo-stage">
          <img v-if="activePhoto" :src="activePhoto" :alt="match.wine?.name" />
          <span v-else>Фото вина</span>
        </div>
        <div v-if="realPhotos(match.wine).length > 1" class="thumb-strip">
          <button
            v-for="photo in realPhotos(match.wine)"
            :key="photo.id || photo.url"
            type="button"
            :class="{ active: app.selectedPhotos.value[match.wine_id] === photo.url }"
            @click="app.selectedPhotos.value = { ...app.selectedPhotos.value, [match.wine_id]: photo.url }"
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
        <img v-for="photo in generatedPhotos(match.wine)" :key="photo.id || photo.url" :src="photo.url" :alt="match.wine?.name" />
      </div>
    </div>

    <section class="rating-panel">
      <h3>Поставь свою оценку</h3>
      <div class="rating-actions">
        <button v-for="value in 5" :key="value" type="button" @click="app.setReview(match, value)">
          <Wine :size="34" />
          <span>{{ value }}</span>
        </button>
      </div>
    </section>

    <SearchDiagnostics />
  </section>
</template>

<script setup>
import { Heart, Images, Percent, Thermometer, Utensils, Wine } from "@lucide/vue";

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

const activePhoto = computed(() => {
  if (!props.match?.wine) return "";
  return app.selectedPhotos.value[props.match.wine_id] || mainPhoto(props.match.wine);
});
</script>
