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

    <PhotoViewer v-model:index="open" :photos="photos" label="Фото из отзыва" />
  </div>
</template>

<script setup>
defineProps({
  photos: { type: Array, default: () => [] },
});

const open = ref(null);
</script>
