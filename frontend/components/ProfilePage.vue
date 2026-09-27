<template>
  <section class="profile-page">
    <div class="page-heading">
      <p class="breadcrumbs">Главная › Кабинет</p>
      <h1>Кабинет</h1>
      <p>История сканирований, избранное, отзывы и напоминания.</p>
    </div>

    <section v-if="app.user.value" class="profile-settings" aria-label="Настройки профиля">
      <div class="profile-avatar-block">
        <UserAvatar class="profile-avatar" :src="app.user.value.avatar_url" :name="app.user.value.nickname" :frame="profileFrame" />
        <div>
          <form v-if="nicknameEditing" class="nickname-form" @submit.prevent="saveNickname">
            <input
              ref="nicknameInput"
              v-model="nicknameDraft"
              maxlength="24"
              aria-label="Публичный ник"
              placeholder="Ваш ник"
              @keydown.esc="cancelNickname"
            />
            <button class="icon-button small" type="submit" :disabled="nicknameSaving" aria-label="Сохранить ник">
              <LoaderCircle v-if="nicknameSaving" :size="16" class="spin" />
              <Check v-else :size="16" />
            </button>
            <button class="icon-button small" type="button" aria-label="Отменить" @click="cancelNickname">
              <X :size="16" />
            </button>
          </form>
          <strong v-else class="nickname-line">
            {{ app.user.value.nickname || app.user.value.email }}
            <button class="nickname-edit" type="button" aria-label="Изменить ник" title="Изменить ник" @click="editNickname">
              <Pencil :size="14" />
            </button>
          </strong>
          <small v-if="nicknameError" class="profile-settings-message error">{{ nicknameError }}</small>
          <span>{{ app.user.value.email }} · виден только вам</span>
          <span>{{ reviewsTotal }} {{ reviewsWord(reviewsTotal) }}</span>
          <small v-if="avatarError" class="profile-settings-message error">{{ avatarError }}</small>
        </div>
        <input ref="avatarInput" class="visually-hidden" type="file" accept="image/*" @change="uploadAvatar" />
        <button class="ui-button secondary small" type="button" :disabled="avatarUploading" @click="avatarInput?.click()">
          <LoaderCircle v-if="avatarUploading" :size="16" class="spin" />
          <Camera v-else :size="16" />
          Аватар
        </button>
      </div>

      <form class="profile-period-form" @submit.prevent="savePeriod">
        <div class="profile-period-field">
          <span>Период уведомлений по оценкам комментариев</span>
          <div class="profile-period-chips" role="radiogroup" aria-label="Период уведомлений по оценкам комментариев">
            <button
              v-for="option in periodOptions"
              :key="option.value"
              type="button"
              class="profile-period-chip"
              :class="{ selected: periodMinutes === option.value }"
              :aria-checked="periodMinutes === option.value"
              role="radio"
              @click="periodMinutes = option.value"
            >
              {{ option.label }}
            </button>
          </div>
        </div>
        <button class="ui-button primary small" type="submit" :disabled="settingsSaving">
          <LoaderCircle v-if="settingsSaving" :size="16" class="spin" />
          <Check v-else-if="settingsSaved" :size="16" />
          <Save v-else :size="16" />
          {{ settingsSaving ? "Сохраняем" : settingsSaved ? "Сохранено" : "Сохранить" }}
        </button>
        <p v-if="settingsMessage" class="profile-settings-message" :class="{ error: settingsError }">
          {{ settingsMessage }}
        </p>
      </form>
    </section>

    <div class="profile-tabs">
      <button
        v-for="tab in app.profileTabs.value"
        :key="tab.id"
        :class="{ active: app.profileTab.value === tab.id }"
        type="button"
        @click="app.profileTab.value = tab.id"
      >
        {{ tab.label }} <span>{{ tab.count }}</span>
      </button>
    </div>

    <!-- история доступна и без входа (временная, по cookie); остальные разделы — только с аккаунтом -->
    <div v-if="!app.user.value && app.profileTab.value !== 'history'" class="auth-required">
      <h2>Войдите, чтобы открыть этот раздел</h2>
      <p>Избранное, отзывы и напоминания хранятся в аккаунте. История поиска доступна и без входа.</p>
      <button class="ui-button primary" type="button" @click="app.authDialog.value = true">Войти или зарегистрироваться</button>
    </div>

    <div v-else-if="app.profileTab.value === 'history'" class="profile-list">
      <div v-if="!app.user.value" class="guest-banner">
        <Clock3 :size="20" />
        <div>
          <strong>Это временная история</strong>
          <span>Она хранится только в этом браузере и через несколько дней исчезнет. Войдите — и она сохранится в аккаунт.</span>
        </div>
        <button class="ui-button primary small" type="button" @click="app.authDialog.value = true">Войти</button>
      </div>
      <p v-if="!app.history.value.items.length" class="profile-empty">
        История пуста. Отсканируйте этикетку — найденные вина появятся здесь.
      </p>
      <button
        v-for="item in app.history.value.items"
        :key="item.id"
        class="history-item"
        type="button"
        :disabled="!item.top_wine"
        @click="app.openWine(item.top_wine, { searchId: item.id, context: `Скан ${formatDate(item.created_at)}` })"
      >
        <span class="viewed-wine-photo">
          <img v-if="mainPhoto(item.top_wine)" :src="mainPhoto(item.top_wine)" :alt="item.top_wine?.name" />
        </span>
        <span class="viewed-wine-body">
          <strong>{{ item.top_wine?.name || "Поиск без результата" }}</strong>
          <span>{{ [item.top_wine?.producer, categoryLine(item.top_wine)].filter((part) => part && part !== "—").join(" · ") }}</span>
          <small>{{ formatDate(item.created_at) }} · уверенность {{ formatPercent(item.confidence) }}</small>
          <span v-if="reviewOf(item.top_wine)" class="glass-row" aria-label="Ваша оценка">
            <WineGlass v-for="value in 5" :key="value" :filled="value <= (reviewOf(item.top_wine).rating || 0)" :size="18" />
          </span>
        </span>
        <ChevronRight v-if="item.top_wine" :size="20" class="notification-card-chevron" />
      </button>
    </div>

    <div v-else-if="app.profileTab.value === 'favorites'" class="wine-grid">
      <article v-for="item in app.favorites.value.items" :key="item.wine.id" class="favorite-card">
        <button class="favorite-open" type="button" @click="app.openWine(item.wine, { searchId: item.search_id, context: 'В избранном' })">
          <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" :alt="item.wine.name" />
          <h3>{{ item.wine.name }}</h3>
          <p>{{ item.wine.producer }}</p>
        </button>
        <button class="ui-button transparent" type="button" @click="app.removeFavorite(item.wine.id)">Убрать</button>
      </article>
    </div>

    <div v-else-if="app.profileTab.value === 'reviews'" class="profile-list">
      <p v-if="!app.reviews.value.items.length" class="profile-empty">
        Вы ещё не оценивали вина. Оценку можно поставить на карточке вина после сканирования или из напоминания.
      </p>
      <article v-for="item in app.reviews.value.items" :key="item.wine_id" class="review-item">
        <div class="viewed-wine-photo">
          <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" :alt="item.wine?.name" />
        </div>
        <div class="viewed-wine-body">
          <button class="link-button" type="button" @click="app.openWine(item.wine, { context: 'Ваш отзыв' })">
            {{ item.wine?.name }}
          </button>
          <span>{{ item.wine?.producer }}</span>
          <span class="glass-row" :aria-label="`Оценка: ${item.rating || 'без оценки'} из 5`">
            <WineGlass v-for="value in 5" :key="value" :filled="value <= (item.rating || 0)" :size="20" />
          </span>
          <q v-if="item.comment">{{ item.comment }}</q>
          <ReviewPhotos :photos="item.photos" />
          <div v-if="item.comment" class="review-reactions own" aria-label="Реакции на ваш комментарий">
            <span><ThumbsUp :size="16" />{{ item.likes || 0 }}</span>
            <span><ThumbsDown :size="16" />{{ item.dislikes || 0 }}</span>
          </div>
          <small>{{ formatDate(item.updated_at || item.created_at) }}</small>
        </div>
        <button
          class="ui-button tertiary small"
          type="button"
          @click="app.openRating({ wine: item.wine, wineId: item.wine_id, review: item })"
        >
          <Pencil :size="16" />
          Изменить
        </button>
      </article>
    </div>

    <ProfileAchievements v-else-if="app.profileTab.value === 'achievements'" />

    <div v-else class="profile-list">
      <div v-if="app.notifications.value.items.length" class="profile-list-toolbar">
        <span>{{ app.notifications.value.unread ? `Новых: ${app.notifications.value.unread}` : "Все прочитаны" }}</span>
        <div class="profile-list-toolbar-actions">
          <button
            v-if="app.notifications.value.unread"
            class="ui-button transparent small"
            type="button"
            @click="app.markAllNotificationsRead"
          >
            <CheckCheck :size="16" />
            Прочитать все
          </button>
          <button class="ui-button transparent small" type="button" @click="deleteAllNotifications">
            <Trash2 :size="16" />
            Удалить все
          </button>
        </div>
      </div>
      <p v-if="!app.notifications.value.items.length" class="profile-empty">
        Напоминаний пока нет. Через час после поиска мы спросим, взяли ли вы что-нибудь из просмотренного.
      </p>
      <div
        v-for="item in app.notifications.value.items"
        :key="item.id"
        class="notification-card"
        :class="{ unread: !item.read_at }"
      >
        <button class="notification-card-open" type="button" @click="openNotification(item)">
          <span class="notification-icon" :class="item.kind">
            <Trophy v-if="item.kind === 'achievement'" :size="20" />
            <ThumbsUp v-else-if="item.kind === 'review_reactions'" :size="20" />
            <Bell v-else :size="20" />
          </span>
          <span class="notification-card-body">
            <strong>{{ item.message }}</strong>
            <span>{{ formatDate(item.created_at) }} · {{ notificationKind(item) }}</span>
          </span>
          <span v-if="!item.read_at" class="new-badge">Новое</span>
          <ChevronRight :size="20" class="notification-card-chevron" />
        </button>
        <button
          class="notification-delete"
          type="button"
          aria-label="Удалить уведомление"
          title="Удалить"
          @click="app.deleteNotification(item.id)"
        >
          <Trash2 :size="18" />
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { Bell, Camera, Check, CheckCheck, ChevronRight, Clock3, LoaderCircle, Pencil, Save, ThumbsDown, ThumbsUp, Trash2, Trophy, X } from "@lucide/vue";

const app = useWineApp();
const avatarInput = ref(null);
const periodMinutes = ref(24 * 60);
const basePeriodOptions = [
  { label: "1 мин", value: 1 },
  { label: "1 час", value: 60 },
  { label: "24 часа", value: 24 * 60 },
  { label: "7 дней", value: 7 * 24 * 60 },
  { label: "30 дней", value: 30 * 24 * 60 },
];
const settingsSaving = ref(false);
const settingsSaved = ref(false);
const settingsError = ref(false);
const settingsMessage = ref("");
let settingsMessageTimer = null;

const avatarUploading = ref(false);
const avatarError = ref("");
// рамка и счётчик — по комментариям (оценка без текста не считается), как на бэкенде
const commentsCount = computed(() => app.reviews.value.items.filter((item) => item.comment).length);
const profileFrame = computed(() => frameForCount(commentsCount.value));
const reviewsTotal = computed(() => app.reviews.value.items.length);
const periodOptions = computed(() => {
  if (basePeriodOptions.some((option) => option.value === periodMinutes.value)) return basePeriodOptions;
  return [{ label: formatPeriod(periodMinutes.value), value: periodMinutes.value }, ...basePeriodOptions];
});

watch(
  () => app.user.value?.review_notification_period_minutes,
  (value) => {
    periodMinutes.value = value || 24 * 60;
  },
  { immediate: true },
);

function winesWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "вино";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "вина";
  return "вин";
}

function reviewsWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "отзыв";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "отзыва";
  return "отзывов";
}

function frameForCount(count) {
  if (count > 100) return "diamond";
  if (count >= 50) return "gold";
  if (count >= 10) return "silver";
  if (count >= 1) return "bronze";
  return "none";
}

function formatPeriod(minutes) {
  if (minutes < 60) return `${minutes} мин`;
  if (minutes % 1440 === 0) return `${minutes / 1440} дн.`;
  if (minutes % 60 === 0) return `${minutes / 60} ч`;
  return `${minutes} мин`;
}

// --- публичный ник ------------------------------------------------------------------

const nicknameEditing = ref(false);
const nicknameDraft = ref("");
const nicknameSaving = ref(false);
const nicknameError = ref("");
const nicknameInput = ref(null);

async function editNickname() {
  nicknameDraft.value = app.user.value?.nickname || "";
  nicknameError.value = "";
  nicknameEditing.value = true;
  await nextTick();
  nicknameInput.value?.select();
}

function cancelNickname() {
  nicknameEditing.value = false;
  nicknameError.value = "";
}

async function saveNickname() {
  const nickname = nicknameDraft.value.trim();
  if (nickname === app.user.value?.nickname) return cancelNickname();
  nicknameSaving.value = true;
  nicknameError.value = "";
  try {
    await app.updateProfileSettings({ nickname });
    nicknameEditing.value = false;
    app.reviewsVersion.value += 1; // ник в отзывах
    app.loadAchievements().catch(() => {}); // и в рейтинге
  } catch (error) {
    nicknameError.value = error?.data?.detail || "Не удалось сохранить ник";
  } finally {
    nicknameSaving.value = false;
  }
}

async function uploadAvatar(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  avatarUploading.value = true;
  avatarError.value = "";
  try {
    await app.uploadAvatar(file);
  } catch (error) {
    avatarError.value = error?.data?.detail || "Не удалось загрузить аватар";
  } finally {
    avatarUploading.value = false;
  }
}

// напоминание о винах открывает список вин; остальные ведут в свой раздел кабинета
const NOTIFICATION_TABS = { review_reactions: "reviews", achievement: "achievements" };

async function openNotification(item) {
  const tab = NOTIFICATION_TABS[item.kind];
  if (!tab) return app.openNotificationDetail(item.id);
  if (!item.read_at) await app.markNotificationRead(item.id);
  app.profileTab.value = tab;
  if (tab === "reviews") await app.loadReviews();
}

function notificationKind(item) {
  if (item.kind === "achievement") return "достижения";
  if (item.kind === "review_reactions") return "отзывы";
  return `${item.wines_count} ${winesWord(item.wines_count)}`;
}

async function deleteAllNotifications() {
  if (window.confirm("Удалить все уведомления? Восстановить их не получится.")) await app.deleteAllNotifications();
}

async function savePeriod() {
  settingsSaving.value = true;
  settingsSaved.value = false;
  settingsError.value = false;
  settingsMessage.value = "";
  try {
    await app.updateProfileSettings({ review_notification_period_minutes: periodMinutes.value || 1 });
    settingsSaved.value = true;
    settingsMessage.value = "Настройки сохранены";
    resetSettingsMessageLater();
  } catch {
    settingsError.value = true;
    settingsMessage.value = "Не удалось сохранить настройки";
  } finally {
    settingsSaving.value = false;
  }
}

const { categoryLine, formatDate, formatPercent, mainPhoto } = useWineFormat();

function reviewOf(wine) {
  return wine ? app.reviews.value.items.find((item) => item.wine_id === wine.id) : null;
}

function resetSettingsMessageLater() {
  if (settingsMessageTimer) clearTimeout(settingsMessageTimer);
  settingsMessageTimer = setTimeout(() => {
    settingsSaved.value = false;
    settingsMessage.value = "";
    settingsMessageTimer = null;
  }, 2200);
}

onBeforeUnmount(() => {
  if (settingsMessageTimer) clearTimeout(settingsMessageTimer);
});
</script>
