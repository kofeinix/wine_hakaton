<template>
  <section class="placeholder-page">
    <div class="page-heading">
      <p class="breadcrumbs">Главная › Сомелье</p>
      <h1>Сомелье</h1>
      <p>Раздел подготовлен визуально, логика появится позже.</p>
    </div>
    <div class="sommelier-card">
      <Sparkles :size="48" />
      <h2>Здесь будет винный помощник</h2>
      <p>Подбор вина под блюдо, повод и бюджет подключим отдельным сценарием.</p>
    </div>
  </section>
</template>

<script setup>
import { Sparkles } from "@lucide/vue";
</script>
