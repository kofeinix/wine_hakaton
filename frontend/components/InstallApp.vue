<template>
  <section v-if="visible" class="install-app" aria-labelledby="install-app-title">
    <img class="install-app-icon" src="/icon-192.png" alt="" width="56" height="56" />
    <div class="install-app-body">
      <h2 id="install-app-title">Сканер вин на главном экране</h2>
      <p>Открывайте сканер одним касанием — как приложение, без адресной строки.</p>

      <button v-if="installPrompt" class="ui-button primary" type="button" @click="install">
        <Download :size="18" />
        Установить
      </button>

      <ol v-else-if="platform === 'ios'" class="install-app-steps">
        <li>
          Нажмите
          <span class="install-app-key"><Share :size="16" /> Поделиться</span>
          в панели Safari
        </li>
        <li>Выберите <span class="install-app-key"><SquarePlus :size="16" /> На экран «Домой»</span></li>
      </ol>

      <ol v-else class="install-app-steps">
        <li>
          Откройте меню браузера
          <span class="install-app-key"><EllipsisVertical :size="16" /></span>
        </li>
        <li>Выберите <span class="install-app-key">Добавить на главный экран</span> или «Установить приложение»</li>
      </ol>
    </div>
    <button class="install-app-close" type="button" aria-label="Скрыть подсказку" @click="hide">
      <X :size="18" />
    </button>
  </section>
</template>

<script setup>
import { Download, EllipsisVertical, Share, SquarePlus, X } from "@lucide/vue";

const HIDDEN_KEY = "wine.installHidden";

const visible = ref(false);
const platform = ref("android");
const installPrompt = ref(null); // событие Chrome: можно показать системное окно установки

function isStandalone() {
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}

function isPhone() {
  // только телефоны и планшеты: сенсорный экран без точного указателя
  return window.matchMedia("(pointer: coarse)").matches && window.matchMedia("(max-width: 1024px)").matches;
}

function onBeforeInstall(event) {
  event.preventDefault(); // своя кнопка вместо баннера браузера
  installPrompt.value = event;
}

function onInstalled() {
  visible.value = false;
  installPrompt.value = null;
}

async function install() {
  const prompt = installPrompt.value;
  if (!prompt) return;
  prompt.prompt();
  const { outcome } = await prompt.userChoice;
  installPrompt.value = null;
  if (outcome === "accepted") visible.value = false;
}

function hide() {
  visible.value = false;
  try {
    localStorage.setItem(HIDDEN_KEY, "1");
  } catch {
    // без хранилища просто скрываем до перезагрузки
  }
}

onMounted(() => {
  window.addEventListener("beforeinstallprompt", onBeforeInstall);
  window.addEventListener("appinstalled", onInstalled);
  let hidden = false;
  try {
    hidden = localStorage.getItem(HIDDEN_KEY) === "1";
  } catch {
    hidden = false;
  }
  const ua = navigator.userAgent;
  // iPad с iPadOS представляется как Mac, но с сенсорным экраном
  platform.value = /iphone|ipad|ipod/i.test(ua) || (/macintosh/i.test(ua) && navigator.maxTouchPoints > 1) ? "ios" : "android";
  visible.value = !hidden && !isStandalone() && isPhone();
});

onBeforeUnmount(() => {
  window.removeEventListener("beforeinstallprompt", onBeforeInstall);
  window.removeEventListener("appinstalled", onInstalled);
});
</script>
