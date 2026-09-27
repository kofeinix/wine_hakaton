<template>
  <div v-if="photos?.length" class="review-photos">
    <button
      v-for="(photo, index) in photos"
      :key="photo.id"
      class="review-photo-thumb"
      type="button"
      :aria-label="`Открыть фото ${index + 1} из ${photos.length}`"
      @click="open = index"
    >
      <img :src="photo.url" alt="" loading="lazy" />
    </button>

    <Teleport to="body">
      <div v-if="open !== null" class="photo-viewer" role="dialog" aria-label="Фото из отзыва" @click.self="open = null">
        <button class="close-button" type="button" aria-label="Закрыть" @click="open = null">
          <X :size="20" />
        </button>
        <button v-if="photos.length > 1" class="photo-viewer-nav prev" type="button" aria-label="Предыдущее фото" @click="step(-1)">
          <ChevronLeft :size="24" />
        </button>
        <img :src="photos[open].url" alt="" />
        <button v-if="photos.length > 1" class="photo-viewer-nav next" type="button" aria-label="Следующее фото" @click="step(1)">
          <ChevronRight :size="24" />
        </button>
      </div>
    </Teleport>
  </div>
</template>

<script setup>
import { ChevronLeft, ChevronRight, X } from "@lucide/vue";

const props = defineProps({
  photos: { type: Array, default: () => [] },
});

const open = ref(null);

function step(delta) {
  open.value = (open.value + delta + props.photos.length) % props.photos.length;
}

function onKey(event) {
  if (open.value === null) return;
  if (event.key === "Escape") open.value = null;
  else if (event.key === "ArrowLeft") step(-1);
  else if (event.key === "ArrowRight") step(1);
}

onMounted(() => window.addEventListener("keydown", onKey));
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));
</script>
