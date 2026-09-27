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
        <UserAvatar :src="mine.avatar_url" :name="app.user.value?.email" :frame="mine.author_frame" />
        <div>
          <strong>Ваш отзыв</strong>
          <small>{{ formatDate(mine.updated_at) }}</small>
        </div>
        <button class="ui-button tertiary small review-edit" type="button" aria-label="Изменить отзыв" @click="edit">
          <Pencil :size="16" />
          <span>Изменить</span>
        </button>
      </div>
      <span class="glass-row" :aria-label="`Ваша оценка: ${mine.rating || 'без оценки'} из 5`">
        <WineGlass v-for="value in 5" :key="value" :filled="value <= (mine.rating || 0)" :size="22" />
      </span>
      <p v-if="mine.comment">{{ mine.comment }}</p>
      <div v-if="mine.comment" class="review-reactions own" title="Свой комментарий оценить нельзя" aria-label="Реакции на ваш комментарий">
        <span><ThumbsUp :size="16" />{{ mine.likes }}</span>
        <span><ThumbsDown :size="16" />{{ mine.dislikes }}</span>
      </div>
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
      <article v-for="item in shownOthers" :key="item.user_id" class="review-card">
        <div class="review-card-head">
          <UserAvatar :src="item.avatar_url" :name="item.author" :frame="item.author_frame" />
          <div>
            <strong>{{ item.author }}</strong>
            <small>{{ formatDate(item.updated_at) }}</small>
          </div>
        </div>
        <span v-if="item.rating" class="glass-row" :aria-label="`Оценка ${item.rating} из 5`">
          <WineGlass v-for="value in 5" :key="value" :filled="value <= item.rating" :size="20" />
        </span>
        <p v-if="item.comment">{{ item.comment }}</p>
        <div v-if="item.comment" class="review-reactions" role="group" aria-label="Оценить комментарий">
          <button
            type="button"
            :class="{ active: item.my_reaction === 1 }"
            :aria-pressed="item.my_reaction === 1"
            :disabled="reacting === item.user_id"
            aria-label="Полезный комментарий"
            @click="react(item, 1)"
          >
            <ThumbsUp :size="16" />{{ item.likes }}
          </button>
          <button
            type="button"
            :class="{ active: item.my_reaction === -1 }"
            :aria-pressed="item.my_reaction === -1"
            :disabled="reacting === item.user_id"
            aria-label="Бесполезный комментарий"
            @click="react(item, -1)"
          >
            <ThumbsDown :size="16" />{{ item.dislikes }}
          </button>
        </div>
      </article>
      <button v-if="others.length > shownOthers.length" class="ui-button transparent small" type="button" @click="expanded = true">
        Показать все отзывы ({{ others.length }})
      </button>
    </div>
  </section>
</template>

<script setup>
import { Pencil, ThumbsDown, ThumbsUp } from "@lucide/vue";

const props = defineProps({
  wine: { type: Object, required: true },
});

const app = useWineApp();
const { formatDate } = useWineFormat();
const data = ref(null);
const error = ref("");
const hover = ref(0);
const expanded = ref(false);
const reacting = ref(null);

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

// повторный клик по своей реакции снимает её
async function react(item, value) {
  reacting.value = item.user_id;
  error.value = "";
  try {
    if (item.my_reaction === value) await app.clearReviewReaction(props.wine.id, item.user_id);
    else await app.reactToReview(props.wine.id, item.user_id, value);
  } catch {
    error.value = "Не удалось сохранить реакцию";
  } finally {
    reacting.value = null;
  }
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
