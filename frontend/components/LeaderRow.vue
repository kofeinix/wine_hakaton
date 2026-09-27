<template>
  <li :class="{ me: entry.is_me }">
    <span class="leader-rank" :class="medal">{{ entry.rank }}</span>
    <UserAvatar :src="entry.avatar_url" :name="entry.display_name" />
    <span class="leader-name">
      {{ entry.display_name }}
      <small v-if="entry.is_me">это вы</small>
    </span>
    <span class="leader-count"><Trophy :size="16" /> {{ entry.earned_count }}</span>
  </li>
</template>

<script setup>
import { Trophy } from "@lucide/vue";

const props = defineProps({
  entry: { type: Object, required: true },
});

const medal = computed(() => ({ 1: "gold", 2: "silver", 3: "bronze" })[props.entry.rank] || "");
</script>
