<template>
  <button class="wine-card" type="button" @click="$emit('select')">
    <span v-if="match.wine?.rating" class="rating-badge">
      🍷 {{ Number(match.wine.rating).toFixed(2).replace(/\.00$/, "") }}
    </span>
    <img v-if="mainPhoto(match.wine)" :src="mainPhoto(match.wine)" :alt="match.wine?.name" />
    <span v-else class="no-photo">Фото</span>
    <strong>{{ match.wine?.name || match.slug }}</strong>
    <small>{{ match.wine?.producer }}</small>
    <em>{{ formatPercent(match.final_score) }}</em>
  </button>
</template>

<script setup>
defineProps({
  match: {
    type: Object,
    required: true,
  },
});

defineEmits(["select"]);

const { formatPercent, mainPhoto } = useWineFormat();
</script>
