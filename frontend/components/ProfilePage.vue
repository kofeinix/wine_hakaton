<template>
  <section class="profile-page">
    <div class="page-heading">
      <p class="breadcrumbs">Главная › Кабинет</p>
      <h1>Кабинет</h1>
      <p>История сканирований, избранное, отзывы и напоминания.</p>
    </div>

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

    <div v-else class="profile-list">
      <div v-if="app.notifications.value.unread" class="profile-list-toolbar">
        <span>Новых: {{ app.notifications.value.unread }}</span>
        <button class="ui-button transparent small" type="button" @click="app.markAllNotificationsRead">
          <CheckCheck :size="16" />
          Отметить все прочитанными
        </button>
      </div>
      <p v-if="!app.notifications.value.items.length" class="profile-empty">
        Напоминаний пока нет. Через час после поиска мы спросим, взяли ли вы что-нибудь из просмотренного.
      </p>
      <button
        v-for="item in app.notifications.value.items"
        :key="item.id"
        class="notification-card"
        :class="{ unread: !item.read_at }"
        type="button"
        @click="app.openNotificationDetail(item.id)"
      >
        <span class="notification-icon"><Bell :size="20" /></span>
        <span class="notification-card-body">
          <strong>{{ item.message }}</strong>
          <span>{{ formatDate(item.created_at) }} · {{ item.wines_count }} {{ winesWord(item.wines_count) }}</span>
        </span>
        <span v-if="!item.read_at" class="new-badge">Новое</span>
        <ChevronRight :size="20" class="notification-card-chevron" />
      </button>
    </div>
  </section>
</template>

<script setup>
import { Bell, CheckCheck, ChevronRight, Clock3, Pencil } from "@lucide/vue";

const app = useWineApp();

function winesWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "вино";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "вина";
  return "вин";
}
const { categoryLine, formatDate, formatPercent, mainPhoto } = useWineFormat();

function reviewOf(wine) {
  return wine ? app.reviews.value.items.find((item) => item.wine_id === wine.id) : null;
}
</script>
