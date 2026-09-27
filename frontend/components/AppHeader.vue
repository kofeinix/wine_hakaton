<template>
  <header ref="header" class="site-header" :class="{ 'menu-open': menuOpen }">
    <a class="brand" href="/" aria-label="Сканер вин" @click.prevent="go('scanner')">
      <img class="brand-logo" src="/logo-asd.png" alt="ASD Hackathon Team" width="58" height="48" />
      <span>
        <strong>Сканер вин</strong>
        <small class="brand-team">by ASD team</small>
        <small>по мотивам Своё Вино</small>
      </span>
    </a>

    <nav id="main-nav" class="main-nav" aria-label="Основная навигация">
      <button :class="{ active: app.activeView.value === 'scanner' }" type="button" @click="go('scanner')">
        Сканер вин
      </button>
      <button :class="{ active: app.activeView.value === 'sommelier' }" type="button" @click="go('sommelier')">
        Сомелье
      </button>
      <a href="https://vino-svoe.ru" target="_blank" rel="noreferrer" @click="menuOpen = false">Портал Своё Вино</a>
      <button
        v-if="app.user.value"
        class="mobile-only"
        :class="{ active: app.activeView.value === 'profile' }"
        type="button"
        @click="go('profile')"
      >
        Кабинет · {{ app.user.value.email }}
      </button>
      <button v-else class="ui-button secondary mobile-only" type="button" @click="login">Войти</button>
    </nav>

    <div class="header-actions">
      <button class="icon-button" type="button" aria-label="Уведомления" @click="notifications">
        <Bell :size="20" />
        <span v-if="app.notifications.value.unread" class="dot">{{ app.notifications.value.unread }}</span>
      </button>
      <button v-if="app.user.value" class="pill-user desktop-only" type="button" @click="go('profile')">
        <span v-if="app.user.value.avatar_url" class="header-avatar"><img :src="app.user.value.avatar_url" alt="" /></span>
        <UserRound v-else :size="18" />
        <span>{{ app.user.value.email }}</span>
      </button>
      <button v-else class="ui-button secondary small desktop-only" type="button" @click="login">Войти</button>
      <button
        class="menu-button"
        type="button"
        :aria-label="menuOpen ? 'Закрыть меню' : 'Открыть меню'"
        :aria-expanded="menuOpen"
        aria-controls="main-nav"
        @click="menuOpen = !menuOpen"
      >
        <X v-if="menuOpen" :size="22" />
        <Menu v-else :size="22" />
      </button>
    </div>
  </header>
</template>

<script setup>
import { Bell, Menu, UserRound, X } from "@lucide/vue";

const app = useWineApp();
const menuOpen = ref(false);
const header = ref(null);

function go(view) {
  app.activeView.value = view;
  menuOpen.value = false;
}

function login() {
  menuOpen.value = false;
  app.authDialog.value = true;
}

function notifications() {
  menuOpen.value = false;
  app.openNotifications();
}

function onPointerDown(event) {
  if (menuOpen.value && !header.value?.contains(event.target)) menuOpen.value = false;
}

function onKeydown(event) {
  if (event.key === "Escape") menuOpen.value = false;
}

onMounted(() => {
  document.addEventListener("pointerdown", onPointerDown);
  document.addEventListener("keydown", onKeydown);
});

onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onPointerDown);
  document.removeEventListener("keydown", onKeydown);
});
</script>
