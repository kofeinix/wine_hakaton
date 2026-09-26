<template>
  <aside class="side-panel">
    <div class="stat-row">
      <span>История</span>
      <strong>{{ app.history.value.items.length }}</strong>
    </div>
    <div class="stat-row">
      <span>Избранное</span>
      <strong>{{ app.favorites.value.items.length }}</strong>
    </div>
    <div class="stat-row">
      <span>Уведомления</span>
      <strong>{{ app.notifications.value.unread }}</strong>
    </div>
    <button class="ui-button secondary" type="button" @click="app.activeView.value = 'profile'">Открыть кабинет</button>
  </aside>
</template>

<script setup>
const app = useWineApp();
</script>
