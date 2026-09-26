<template>
  <div v-if="detail" class="modal-backdrop" @click.self="close">
    <div class="notification-modal">
      <button class="close-button" type="button" aria-label="Закрыть" @click="close">
        <X :size="20" />
      </button>

      <header class="notification-modal-head">
        <span class="notification-icon"><Bell :size="20" /></span>
        <div>
          <h2>{{ detail.message }}</h2>
          <p>{{ period(detail) }} · {{ detail.wines.length }} {{ winesWord(detail.wines.length) }}</p>
        </div>
      </header>

      <p class="notification-modal-lead">Отметьте, что понравилось: вино попадёт в избранное, а оценка поможет советовать точнее.</p>

      <div class="viewed-list">
        <article v-for="item in detail.wines" :key="item.wine.id" class="viewed-wine">
          <button
            class="viewed-wine-photo photo-button"
            type="button"
            :aria-label="`Открыть карточку: ${item.wine.name}`"
            @click="open(item)"
          >
            <img v-if="mainPhoto(item.wine)" :src="mainPhoto(item.wine)" alt="" />
          </button>

          <div class="viewed-wine-body">
            <button class="link-button" type="button" @click="open(item)">{{ item.wine.name }}</button>
            <span>{{ [item.wine.producer, categoryLine(item.wine)].filter((part) => part && part !== "—").join(" · ") }}</span>
            <small>Смотрели {{ formatDate(item.viewed_at) }}</small>

            <div v-if="item.review" class="viewed-review">
              <span class="glass-row" :aria-label="`Ваша оценка: ${item.review.rating || 'без оценки'} из 5`">
                <WineGlass v-for="value in 5" :key="value" :filled="value <= (item.review.rating || 0)" :size="20" />
              </span>
              <q v-if="item.review.comment">{{ item.review.comment }}</q>
            </div>
          </div>

          <div class="viewed-wine-actions">
            <span v-if="item.is_favorite" class="status-chip">
              <Heart :size="16" fill="currentColor" />
              В избранном
            </span>
            <button
              v-else
              class="ui-button secondary small"
              type="button"
              @click="app.addFavorite({ wine: item.wine, wine_id: item.wine.id }, item.search_id)"
            >
              <Heart :size="16" />
              В избранное
            </button>
            <button
              class="ui-button small"
              :class="item.review ? 'tertiary' : 'primary'"
              type="button"
              @click="app.openRating({ wine: item.wine, notificationId: detail.id, review: item.review })"
            >
              <WineIcon v-if="!item.review" :size="16" />
              <Pencil v-else :size="16" />
              {{ item.review ? "Изменить отзыв" : "Оценить" }}
            </button>
          </div>
        </article>
      </div>
    </div>
  </div>
</template>

<script setup>
import { Bell, Heart, Pencil, Wine as WineIcon, X } from "@lucide/vue";

const app = useWineApp();
const { categoryLine, formatDate, mainPhoto } = useWineFormat();
const detail = computed(() => app.notificationDetail.value);

function open(item) {
  app.openWine(item.wine, { searchId: item.search_id, context: `Смотрели ${formatDate(item.viewed_at)}` });
}

function close() {
  app.notificationDetail.value = null;
}

function winesWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "вино";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "вина";
  return "вин";
}

function period(item) {
  const start = formatDate(item.period_start);
  const end = formatDate(item.period_end);
  return start === end ? start : `${start} — ${end}`;
}
</script>
