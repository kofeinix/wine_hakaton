<template>
  <section class="scanner-page">
    <div class="page-heading">
      <p class="breadcrumbs">Главная › Сканер вин</p>
      <h1>Найдите вино по этикетке</h1>
      <p>Сфотографируйте бутылку или загрузите снимок. Мы покажем лучшее совпадение, похожие вина и подскажем блюда.</p>
    </div>

    <section class="scanner-layout">
      <ScannerCard />
      <ScannerStats />
    </section>

    <BottlePicker />

    <WineResult :match="app.bestMatch.value" />

    <section v-if="app.similarMatches.value.length" class="similar-section">
      <div class="section-title">
        <h2>Похожие варианты</h2>
      </div>
      <div class="wine-grid">
        <WineCard
          v-for="match in app.similarMatches.value"
          :key="match.wine_id"
          :match="match"
          @select="openSimilar(match)"
        />
      </div>
    </section>
  </section>
</template>

<script setup>
const app = useWineApp();
const { formatPercent } = useWineFormat();

// похожее вино — отдельным окном, основной результат не трогаем
function openSimilar(match) {
  app.openWine(match.wine, {
    searchId: app.searchResponse.value?.search_id,
    context: `Похожий вариант · совпадение ${formatPercent(match.similarity)}`,
  });
}
</script>
