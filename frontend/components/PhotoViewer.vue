<template>
  <Teleport to="body">
    <div
      v-if="index !== null && photos[index]"
      class="photo-viewer"
      role="dialog"
      aria-modal="true"
      :aria-label="label"
      @click.self="close"
      @touchstart.passive="touchStart"
      @touchend.passive="touchEnd"
    >
      <button class="close-button" type="button" aria-label="Закрыть" @click="close">
        <X :size="20" />
      </button>
      <button v-if="photos.length > 1" class="photo-viewer-nav prev" type="button" aria-label="Предыдущее фото" @click="step(-1)">
        <ChevronLeft :size="24" />
      </button>
      <figure>
        <img :src="photos[index].url" :alt="photos[index].caption || ''" />
        <figcaption v-if="photos.length > 1 || photos[index].caption">
          <span v-if="photos.length > 1">{{ index + 1 }} / {{ photos.length }}</span>
          <span v-if="photos[index].caption">{{ photos[index].caption }}</span>
        </figcaption>
      </figure>
      <button v-if="photos.length > 1" class="photo-viewer-nav next" type="button" aria-label="Следующее фото" @click="step(1)">
        <ChevronRight :size="24" />
      </button>
    </div>
  </Teleport>
</template>

<script setup>
import { ChevronLeft, ChevronRight, X } from "@lucide/vue";

// photos: [{ url, caption? }]; index — открытое фото или null (закрыто)
const props = defineProps({
  photos: { type: Array, default: () => [] },
  index: { type: Number, default: null },
  label: { type: String, default: "Фото" },
});
const emit = defineEmits(["update:index"]);

let touchX = null;

function close() {
  emit("update:index", null);
}

function step(delta) {
  emit("update:index", (props.index + delta + props.photos.length) % props.photos.length);
}

function touchStart(event) {
  touchX = event.touches[0]?.clientX ?? null;
}

// свайп влево/вправо листает
function touchEnd(event) {
  if (touchX === null || props.photos.length < 2) return;
  const dx = (event.changedTouches[0]?.clientX ?? touchX) - touchX;
  touchX = null;
  if (Math.abs(dx) > 40) step(dx < 0 ? 1 : -1);
}

function onKey(event) {
  if (props.index === null) return;
  if (event.key === "Escape") close();
  else if (event.key === "ArrowLeft") step(-1);
  else if (event.key === "ArrowRight") step(1);
}

onMounted(() => window.addEventListener("keydown", onKey));
onBeforeUnmount(() => window.removeEventListener("keydown", onKey));
</script>
