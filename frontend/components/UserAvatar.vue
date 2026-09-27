<template>
  <span class="review-avatar" :class="`frame-${frame || 'none'}`" :title="frameTitle">
    <span class="review-avatar-media">
      <img v-if="src" :src="src" alt="" />
      <span v-else>{{ initial }}</span>
    </span>
  </span>
</template>

<script setup>
const props = defineProps({
  src: { type: String, default: null },
  name: { type: String, default: "" },
  frame: { type: String, default: "none" },
});

const FRAME_TITLES = {
  bronze: "Бронзовая рамка: от 1 комментария",
  silver: "Серебряная рамка: от 10 комментариев",
  gold: "Золотая рамка: от 50 комментариев",
  diamond: "Бриллиантовая рамка: больше 100 комментариев",
};

const initial = computed(() => (props.name || "П").slice(0, 1).toUpperCase());
const frameTitle = computed(() => FRAME_TITLES[props.frame] || undefined);
</script>
