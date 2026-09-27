<template>
  <div v-if="detail" class="modal-backdrop" @click.self="close">
    <article class="wine-detail-modal" aria-labelledby="wine-detail-title">
      <button class="close-button" type="button" aria-label="Закрыть" @click="close">
        <X :size="20" />
      </button>

      <div class="wine-detail-top">
        <button
          class="wine-detail-photo"
          type="button"
          :disabled="!photos.length"
          :aria-label="photos.length ? `Открыть фото: ${wine.name}` : undefined"
          @click="openViewer(photos, 0)"
        >
          <img v-if="mainPhoto(wine)" :src="mainPhoto(wine)" :alt="wine.name" />
          <span v-if="photos.length > 1" class="wine-detail-photo-count">
            <Images :size="14" />
            {{ photos.length }}
          </span>
        </button>
        <div class="wine-detail-main">
          <p v-if="detail.context" class="wine-detail-context">{{ detail.context }}</p>
          <h2 id="wine-detail-title"><TermText :text="wine.name" /></h2>
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

      <p v-if="wine.description" class="wine-detail-description"><TermText :text="wine.description" /></p>

      <section v-if="photos.length > 1 || scenes.length" class="wine-gallery" aria-labelledby="wine-gallery-title">
        <div class="wine-gallery-head">
          <h3 id="wine-gallery-title">Галерея</h3>
          <div v-if="scenes.length" class="sm-chips" role="tablist" aria-label="Что показать">
            <button
              type="button"
              role="tab"
              class="sm-chip small"
              :class="{ selected: galleryTab === 'photos' }"
              :aria-selected="galleryTab === 'photos'"
              @click="galleryTab = 'photos'"
            >
              Фото <span>{{ photos.length }}</span>
            </button>
            <button
              type="button"
              role="tab"
              class="sm-chip small"
              :class="{ selected: galleryTab === 'scenes' }"
              :aria-selected="galleryTab === 'scenes'"
              @click="galleryTab = 'scenes'"
            >
              Сцены <span>{{ scenes.length }}</span>
            </button>
          </div>
        </div>
        <p v-if="galleryTab === 'scenes'" class="wine-gallery-note">
          Сцены сгенерированы по фото бутылки — так её легче узнать на полке и на столе.
        </p>
        <div class="wine-gallery-grid">
          <button
            v-for="(photo, index) in shownGallery"
            :key="photo.url"
            class="wine-gallery-tile"
            type="button"
            :aria-label="`Открыть фото ${index + 1} из ${gallery.length}`"
            @click="openViewer(gallery, index)"
          >
            <img :src="photo.url" alt="" loading="lazy" />
            <span v-if="index === GALLERY_PREVIEW - 1 && hiddenCount" class="wine-gallery-more">+{{ hiddenCount }}</span>
          </button>
        </div>
      </section>

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

    <PhotoViewer v-model:index="viewerIndex" :photos="viewerPhotos" :label="wine.name" />
  </div>
</template>

<script setup>
import { Heart, Images, Percent, Thermometer, Wine as WineIcon, X } from "@lucide/vue";

const GALLERY_PREVIEW = 8; // остальные — через плитку «+N» и листание в просмотре

const app = useWineApp();
const {
  alcoholLine,
  categoryLine,
  foodPairing,
  generatedPhotos,
  grapeLine,
  mainPhoto,
  realPhotos,
  regionLine,
  servingTemperature,
  shadeLine,
} = useWineFormat();

const detail = computed(() => app.wineDetail.value);
const wine = computed(() => detail.value?.wine || {});
const isFavorite = computed(() => app.favorites.value.items.some((item) => item.wine?.id === wine.value.id));

const photos = computed(() => realPhotos(wine.value).map((photo) => ({ url: photo.url })));
const scenes = computed(() =>
  generatedPhotos(wine.value).map((photo) => ({ url: photo.url, caption: "Сцена сгенерирована по фото бутылки" })),
);
const galleryTab = ref("photos");
const gallery = computed(() => (galleryTab.value === "scenes" ? scenes.value : photos.value));
const shownGallery = computed(() => gallery.value.slice(0, GALLERY_PREVIEW));
const hiddenCount = computed(() => Math.max(0, gallery.value.length - GALLERY_PREVIEW));
const viewerPhotos = ref([]);
const viewerIndex = ref(null);

function openViewer(list, index) {
  if (!list.length) return;
  viewerPhotos.value = list;
  viewerIndex.value = index;
}

// новое вино — галерея с начала
watch(
  () => wine.value.id,
  () => {
    galleryTab.value = "photos";
    viewerIndex.value = null;
  },
);

function close() {
  app.wineDetail.value = null;
}
</script>
