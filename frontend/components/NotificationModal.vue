<template>
  <div v-if="app.notificationDetail.value" class="modal-backdrop" @click.self="app.notificationDetail.value = null">
    <div class="notification-modal">
      <button class="close-button" type="button" aria-label="Закрыть" @click="app.notificationDetail.value = null">
        <X :size="20" />
      </button>
      <h2>{{ app.notificationDetail.value.message }}</h2>
      <article v-for="item in app.notificationDetail.value.wines" :key="item.wine.id" class="reminder-wine">
        <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" :alt="item.wine.name" />
        <div>
          <strong>{{ item.wine.name }}</strong>
          <span>{{ item.wine.producer }}</span>
          <div class="reminder-actions">
            <button class="ui-button secondary small" type="button" @click="app.addFavorite({ wine: item.wine, wine_id: item.wine.id }, item.search_id)">
              В избранное
            </button>
            <button class="ui-button tertiary small" type="button" @click="app.setReview({ wine: item.wine, wine_id: item.wine.id }, 5, app.notificationDetail.value.id)">
              Оценить на 5
            </button>
          </div>
        </div>
      </article>
    </div>
  </div>
</template>

<script setup>
import { X } from "@lucide/vue";

const app = useWineApp();
const { mainPhoto } = useWineFormat();
</script>
