<template>
  <div v-if="app.authDialog.value" class="modal-backdrop auth-backdrop" @click.self="close">
    <form class="auth-modal" @submit.prevent="app.submitAuth">
      <button class="close-button" type="button" aria-label="Закрыть" @click="close">
        <X :size="20" />
      </button>
      <h2>{{ app.authMode.value === "login" ? "Вход" : "Регистрация" }}</h2>
      <label>
        Email
        <input v-model="app.authForm.value.email" type="email" autocomplete="email" required />
      </label>
      <label>
        Пароль
        <input v-model="app.authForm.value.password" type="password" autocomplete="current-password" required minlength="6" />
      </label>
      <p v-if="app.authError.value" class="error-message">{{ app.authError.value }}</p>
      <button class="ui-button primary large" type="submit">
        {{ app.authMode.value === "login" ? "Войти" : "Создать аккаунт" }}
      </button>
      <button class="ui-button transparent" type="button" @click="toggleMode">
        {{ app.authMode.value === "login" ? "Создать аккаунт" : "Уже есть аккаунт" }}
      </button>
    </form>
  </div>
</template>

<script setup>
import { X } from "@lucide/vue";

const app = useWineApp();

// закрыли окно входа — отложенное действие гостя отменяется
function close() {
  app.authDialog.value = false;
  app.pendingAction.value = null;
}

function toggleMode() {
  app.authMode.value = app.authMode.value === "login" ? "register" : "login";
}
</script>
