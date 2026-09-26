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

    <div v-if="!app.user.value" class="auth-required">
      <h2>Войдите, чтобы открыть личные разделы</h2>
      <p>Анонимная история поиска сохраняется временно и перенесётся в аккаунт после входа.</p>
      <button class="ui-button primary" type="button" @click="app.authDialog.value = true">Войти или зарегистрироваться</button>
    </div>

    <div v-else-if="app.profileTab.value === 'history'" class="profile-list">
      <article v-for="item in app.history.value.items" :key="item.id" class="history-item">
        <img v-if="mainPhoto(item.top_wine)" :src="mainPhoto(item.top_wine)" :alt="item.top_wine?.name" />
        <div>
          <strong>{{ item.top_wine?.name || "Поиск без результата" }}</strong>
          <span>{{ formatDate(item.created_at) }} · уверенность {{ formatPercent(item.confidence) }}</span>
        </div>
      </article>
    </div>

    <div v-else-if="app.profileTab.value === 'favorites'" class="wine-grid">
      <article v-for="item in app.favorites.value.items" :key="item.wine.id" class="favorite-card">
        <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" :alt="item.wine.name" />
        <h3>{{ item.wine.name }}</h3>
        <p>{{ item.wine.producer }}</p>
        <button class="ui-button transparent" type="button" @click="app.removeFavorite(item.wine.id)">Убрать</button>
      </article>
    </div>

    <div v-else-if="app.profileTab.value === 'reviews'" class="profile-list">
      <article v-for="item in app.reviews.value.items" :key="item.wine_id" class="history-item">
        <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" :alt="item.wine?.name" />
        <div>
          <strong>{{ item.wine?.name }}</strong>
          <span>Оценка: {{ item.rating || "—" }}</span>
          <p v-if="item.comment">{{ item.comment }}</p>
        </div>
      </article>
    </div>

    <div v-else class="profile-list">
      <article v-for="item in app.notifications.value.items" :key="item.id" class="notification-item">
        <div>
          <strong>{{ item.message }}</strong>
          <span>{{ formatDate(item.created_at) }} · {{ item.wines_count }} вин</span>
        </div>
        <button class="ui-button secondary small" type="button" @click="app.openNotificationDetail(item.id)">Открыть</button>
      </article>
    </div>
  </section>
</template>

<script setup>
const app = useWineApp();
const { formatDate, formatPercent, mainPhoto } = useWineFormat();
</script>
