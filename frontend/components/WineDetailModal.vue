<template>
  <div v-if="detail" class="modal-backdrop" @click.self="close">
    <article class="wine-detail-modal" aria-labelledby="wine-detail-title">
      <button class="close-button" type="button" aria-label="Закрыть" @click="close">
        <X :size="20" />
      </button>

      <div class="wine-detail-top">
        <div class="wine-detail-photo">
          <img v-if="mainPhoto(wine)" :src="mainPhoto(wine)" :alt="wine.name" />
        </div>
        <div class="wine-detail-main">
          <p v-if="detail.context" class="wine-detail-context">{{ detail.context }}</p>
          <h2 id="wine-detail-title">{{ wine.name }}</h2>
          <a v-if="wine.producer" class="producer-link" :href="wine.source_url || '#'" target="_blank" rel="noreferrer">
            {{ wine.producer }}
          </a>
          <div class="rating-pill">
            <WineIcon :size="18" />
            <span>Народный рейтинг {{ wine.rating || "—" }}</span>
          </div>
          <div class="wine-detail-tiles">
            <div>
              <Thermometer :size="18" />
              <span>Подача</span>
              <strong>{{ servingTemperature(wine) }}</strong>
            </div>
            <div>
              <Percent :size="18" />
              <span>Крепость</span>
              <strong>{{ alcoholLine(wine) }}</strong>
            </div>
          </div>
        </div>
      </div>

      <div class="facts-grid">
        <WineFact label="Регион" :value="regionLine(wine)" />
        <WineFact label="Виноград" :value="grapeLine(wine)" />
        <WineFact label="Категория" :value="categoryLine(wine)" />
        <WineFact label="Оттенок" :value="shadeLine(wine)" />
        <WineFact class="wide" label="Сочетание с блюдами" :value="foodPairing(wine).join(', ') || '—'" />
      </div>

      <p v-if="wine.description" class="wine-detail-description">{{ wine.description }}</p>

      <WineReviews :wine="wine" />

      <div class="hero-actions">
        <span v-if="isFavorite" class="status-chip">
          <Heart :size="16" fill="currentColor" />
          В избранном
        </span>
        <button v-else class="ui-button primary" type="button" @click="app.addFavorite({ wine, wine_id: wine.id }, detail.searchId)">
          <Heart :size="18" />
          В избранное
        </button>
        <a v-if="wine.source_url" class="ui-button secondary as-link" :href="wine.source_url" target="_blank" rel="noreferrer">
          Открыть на портале
        </a>
      </div>
    </article>
  </div>
</template>

<script setup>
import { Heart, Percent, Thermometer, Wine as WineIcon, X } from "@lucide/vue";

const app = useWineApp();
const {
  alcoholLine,
  categoryLine,
  foodPairing,
  grapeLine,
  mainPhoto,
  regionLine,
  servingTemperature,
  shadeLine,
} = useWineFormat();

const detail = computed(() => app.wineDetail.value);
const wine = computed(() => detail.value?.wine || {});
const isFavorite = computed(() => app.favorites.value.items.some((item) => item.wine?.id === wine.value.id));

function close() {
  app.wineDetail.value = null;
}
</script>
