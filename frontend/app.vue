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
    <AchievementToast />
  </div>
</template>

<script setup>
const app = useWineApp();
const urlSync = useUrlSync();

onMounted(() => {
  urlSync.start();
  app.initialize();
});

onBeforeUnmount(() => urlSync.stop());
</script>
