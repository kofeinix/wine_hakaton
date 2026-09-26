<template>
  <header class="site-header">
    <NuxtLink class="brand" to="/" aria-label="Сканер вин">
      <span class="brand-mark">СВ</span>
      <span>
        <strong>Сканер вин</strong>
        <small>по мотивам Своё Вино</small>
      </span>
    </NuxtLink>

    <nav class="main-nav" aria-label="Основная навигация">
      <button :class="{ active: app.activeView.value === 'scanner' }" type="button" @click="app.activeView.value = 'scanner'">
        Сканер вин
      </button>
      <button :class="{ active: app.activeView.value === 'sommelier' }" type="button" @click="app.activeView.value = 'sommelier'">
        Сомелье
      </button>
      <a href="https://vino-svoe.ru" target="_blank" rel="noreferrer">Портал Своё Вино</a>
    </nav>

    <div class="header-actions">
      <button class="icon-button" type="button" aria-label="Уведомления" @click="app.openNotifications">
        <Bell :size="20" />
        <span v-if="app.notifications.value.unread" class="dot">{{ app.notifications.value.unread }}</span>
      </button>
      <button v-if="app.user.value" class="pill-user" type="button" @click="app.activeView.value = 'profile'">
        <UserRound :size="18" />
        <span>{{ app.user.value.email }}</span>
      </button>
      <button v-else class="ui-button secondary small" type="button" @click="app.authDialog.value = true">Войти</button>
    </div>
  </header>
</template>

<script setup>
import { Bell, UserRound } from "@lucide/vue";

const app = useWineApp();
</script>
