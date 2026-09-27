<template>
  <div class="achievement-toasts" aria-live="polite" @mouseenter="hovered = true" @mouseleave="resume">
    <TransitionGroup name="toast">
      <div
        v-for="toast in shown"
        :key="toast.id"
        class="achievement-toast"
        :class="toast.type"
        role="button"
        tabindex="0"
        :aria-label="`${label(toast)}: ${toast.title}. Открыть достижения`"
        @click="open(toast)"
        @keydown.enter="open(toast)"
      >
        <span class="achievement-toast-icon">
          <Trophy v-if="toast.type !== 'progress'" :size="22" />
          <component :is="categoryIcon(toast.category)" v-else :size="22" />
        </span>
        <span class="achievement-toast-body">
          <small>{{ label(toast) }}</small>
          <strong>{{ toast.title }}</strong>
          <span>{{ toast.description }}</span>
          <span v-if="toast.type === 'progress'" class="achievement-toast-progress">
            <span class="timing-track"><span class="timing-bar" :style="{ width: `${percent(toast)}%` }" /></span>
            {{ toast.progress }}/{{ toast.target }}
          </span>
        </span>
        <button class="achievement-toast-close" type="button" aria-label="Скрыть" @click.stop="app.dismissToast(toast.id)">
          <X :size="16" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup>
import { Trophy, X } from "@lucide/vue";

const VISIBLE = 3; // остальные ждут очереди
const LIFETIME_MS = 5000;
const AFTER_HOVER_MS = 3000;

const app = useWineApp();
const { categoryIcon } = useAchievementFormat();
const timers = new Map();
const hovered = ref(false); // пока курсор над уведомлениями, все ждут

const shown = computed(() => app.achievementToasts.value.slice(0, VISIBLE));

function label(toast) {
  if (toast.code === "several") return "Новые достижения";
  if (toast.type === "earned") return "Достижение получено";
  if (toast.type === "summary") return "Достижения";
  return "Прогресс достижения";
}

function percent(toast) {
  return Math.min(100, Math.round((toast.progress / toast.target) * 100));
}

function schedule(id, delay = LIFETIME_MS) {
  pause(id);
  if (hovered.value) {
    timers.set(id, null); // появилось под курсором — запустим после ухода курсора
    return;
  }
  timers.set(
    id,
    setTimeout(() => app.dismissToast(id), delay),
  );
}

// курсор ушёл — показанные уведомления закрываются вместе
function resume() {
  hovered.value = false;
  for (const toast of shown.value) schedule(toast.id, AFTER_HOVER_MS);
}

function pause(id) {
  clearTimeout(timers.get(id));
  timers.delete(id);
}

function open(toast) {
  app.dismissToast(toast.id);
  app.activeView.value = "profile";
  app.profileTab.value = "achievements";
}

// таймер запускается, когда уведомление появилось на экране, а не когда встало в очередь
watch(hovered, (value) => {
  if (!value) return;
  for (const id of [...timers.keys()]) {
    clearTimeout(timers.get(id));
    timers.set(id, null);
  }
});

watch(
  shown,
  (list) => {
    for (const toast of list) if (!timers.has(toast.id)) schedule(toast.id);
    for (const id of [...timers.keys()]) if (!list.some((toast) => toast.id === id)) pause(id);
  },
  { immediate: true },
);

onBeforeUnmount(() => timers.forEach((timer) => clearTimeout(timer)));
</script>
