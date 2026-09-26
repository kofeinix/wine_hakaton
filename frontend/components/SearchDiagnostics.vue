<template>
  <section class="diagnostics">
    <button class="ui-button secondary" type="button" @click="app.showDiagnostics.value = !app.showDiagnostics.value">
      <Activity :size="18" />
      {{ app.showDiagnostics.value ? "Скрыть диагностику" : "Показать диагностику" }}
    </button>
    <div v-if="app.showDiagnostics.value" class="diagnostics-grid">
      <div>
        <span>OCR</span>
        <strong>{{ response?.ocr?.applied ? "применён" : "не применён" }}</strong>
        <small>{{ response?.ocr?.text || "Текст не распознан" }}</small>
      </div>
      <div>
        <span>Кандидатов</span>
        <strong>{{ response?.search?.candidates || 0 }}</strong>
        <small>{{ (response?.search?.active_views || []).join(", ") || "views не указаны" }}</small>
      </div>
      <div>
        <span>Время</span>
        <strong>{{ Math.round(response?.timings_ms?.total_ms || 0) }} мс</strong>
        <small>OCR {{ Math.round(response?.timings_ms?.ocr_ms || 0) }} мс</small>
      </div>
    </div>
  </section>
</template>

<script setup>
import { Activity } from "@lucide/vue";

const app = useWineApp();
const response = computed(() => app.searchResponse.value);
</script>
