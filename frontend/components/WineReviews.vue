<template>
  <section class="wine-reviews" aria-labelledby="wine-reviews-title">
    <div class="wine-reviews-summary">
      <div>
        <h3 id="wine-reviews-title">Отзывы</h3>
        <p v-if="!data?.count" class="wine-reviews-muted">Пока никто не оценил это вино — будьте первым.</p>
        <p v-else class="wine-reviews-muted">{{ data.count }} {{ plural(data.count, ["отзыв", "отзыва", "отзывов"]) }} пользователей сервиса</p>
      </div>

      <div v-if="data?.average_rating" class="wine-reviews-average">
        <strong>{{ data.average_rating.toFixed(1) }}</strong>
        <span class="glass-row" :aria-label="`Средняя оценка ${data.average_rating} из 5`">
          <WineGlass v-for="value in 5" :key="value" :filled="value <= Math.round(data.average_rating)" :size="22" />
        </span>
        <small>{{ data.rated_count }} {{ plural(data.rated_count, ["оценка", "оценки", "оценок"]) }}</small>
      </div>

      <div v-if="data?.rated_count" class="wine-reviews-bars">
        <div v-for="value in [5, 4, 3, 2, 1]" :key="value" class="wine-reviews-bar" :title="`${value} из 5: ${data.distribution[value] || 0}`">
          <span>{{ value }}</span>
          <div class="timing-track">
            <div class="timing-bar" :style="{ width: `${share(value)}%` }" />
          </div>
          <small>{{ data.distribution[value] || 0 }}</small>
        </div>
      </div>
    </div>

    <!-- свой отзыв или приглашение оценить -->
    <article v-if="mine" class="review-card mine">
      <div class="review-card-head">
        <span class="review-avatar">Вы</span>
        <div>
          <strong>Ваш отзыв</strong>
          <small>{{ formatDate(mine.updated_at) }}</small>
        </div>
        <button class="ui-button tertiary small" type="button" @click="edit">
          <Pencil :size="16" />
          Изменить
        </button>
      </div>
      <span class="glass-row" :aria-label="`Ваша оценка: ${mine.rating || 'без оценки'} из 5`">
        <WineGlass v-for="value in 5" :key="value" :filled="value <= (mine.rating || 0)" :size="22" />
      </span>
      <p v-if="mine.comment">{{ mine.comment }}</p>
    </article>

    <div v-else class="review-invite">
      <strong>Поставь свою оценку</strong>
      <div class="glass-rating" role="group" aria-label="Оценка от 1 до 5" @mouseleave="hover = 0">
        <button
          v-for="value in 5"
          :key="value"
          type="button"
          :aria-label="`Оценить на ${value} из 5`"
          :class="{ active: value <= hover }"
          @mouseenter="hover = value"
          @click="app.openRating({ wine, wineId: wine.id, rating: value })"
        >
          <WineGlass :filled="value <= hover" :size="36" />
        </button>
      </div>
    </div>

    <p v-if="error" class="wine-reviews-muted">{{ error }}</p>

    <div v-if="others.length" class="review-list">
      <article v-for="(item, index) in shownOthers" :key="index" class="review-card">
        <div class="review-card-head">
          <span class="review-avatar">{{ item.author.slice(0, 1) }}</span>
          <div>
            <strong>{{ item.author }}</strong>
            <small>{{ formatDate(item.updated_at) }}</small>
          </div>
        </div>
        <span v-if="item.rating" class="glass-row" :aria-label="`Оценка ${item.rating} из 5`">
          <WineGlass v-for="value in 5" :key="value" :filled="value <= item.rating" :size="20" />
        </span>
        <p v-if="item.comment">{{ item.comment }}</p>
      </article>
      <button v-if="others.length > shownOthers.length" class="ui-button transparent small" type="button" @click="expanded = true">
        Показать все отзывы ({{ others.length }})
      </button>
    </div>
  </section>
</template>

<script setup>
import { Pencil } from "@lucide/vue";

const props = defineProps({
  wine: { type: Object, required: true },
});

const app = useWineApp();
const { formatDate } = useWineFormat();
const data = ref(null);
const error = ref("");
const hover = ref(0);
const expanded = ref(false);

const mine = computed(() => data.value?.items?.find((item) => item.is_mine) || null);
const others = computed(() => (data.value?.items || []).filter((item) => !item.is_mine));
const shownOthers = computed(() => (expanded.value ? others.value : others.value.slice(0, 3)));

function plural(count, [one, few, many]) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}

function share(value) {
  const max = Math.max(1, ...Object.values(data.value?.distribution || {}));
  return ((data.value?.distribution?.[value] || 0) / max) * 100;
}

function edit() {
  app.openRating({ wine: props.wine, wineId: props.wine.id, review: mine.value });
}

async function load() {
  if (!props.wine?.id) return;
  error.value = "";
  try {
    data.value = await app.fetchWineReviews(props.wine.id);
  } catch {
    error.value = "Не удалось загрузить отзывы";
  }
}

// перезагружаем при смене вина, входе/выходе и после сохранения своего отзыва
watch(() => [props.wine?.id, app.user.value?.id, app.reviewsVersion.value], load, { immediate: true });
watch(() => props.wine?.id, () => (expanded.value = false));
</script>
