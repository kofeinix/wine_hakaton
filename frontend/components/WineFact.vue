<template>
  <div class="wine-fact">
    <span>{{ label }}</span>
    <strong>{{ value || "—" }}</strong>
  </div>
</template>

<script setup>
defineProps({
  label: {
    type: String,
    required: true,
  },
  value: {
    type: String,
    default: "",
  },
});
</script>
