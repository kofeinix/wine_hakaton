<template>
  <!-- наклонённый бокал, как в блоке оценки vino-svoe; filled — налито вино -->
  <svg class="wine-glass" :class="{ filled }" :width="size" :height="size" viewBox="0 0 32 32" aria-hidden="true">
    <defs>
      <clipPath :id="clipId">
        <path d="M10 4h12c0 7.5-1.8 12.2-6 13.2C11.8 16.2 10 11.5 10 4z" />
      </clipPath>
    </defs>
    <g transform="rotate(-16 16 16)">
      <rect class="wine-glass-wine" x="8" y="8.5" width="16" height="10" :clip-path="`url(#${clipId})`" />
      <path class="wine-glass-line" d="M10 4h12c0 7.5-1.8 12.2-6 13.2C11.8 16.2 10 11.5 10 4z" />
      <path class="wine-glass-line" d="M16 17.2V26.5M11.5 27.5h9" />
    </g>
  </svg>
</template>

<script setup>
defineProps({
  filled: { type: Boolean, default: false },
  size: { type: Number, default: 32 },
});

const clipId = `glass-${useId()}`;
</script>
