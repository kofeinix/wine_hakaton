<template>
  <div class="app-shell">
    <AppHeader />

    <main>
      <ScannerPage v-if="app.activeView.value === 'scanner'" />
      <SommelierPage v-else-if="app.activeView.value === 'sommelier'" />
      <ProfilePage v-else />
    </main>

    <AppFooter />
    <AuthModal />
    <NotificationModal />
    <WineDetailModal />
    <RatingModal />
  </div>
</template>

<script setup>
const app = useWineApp();

onMounted(() => {
  app.initialize();
});
</script>
