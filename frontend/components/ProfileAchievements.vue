<template>
  <div class="achievements-tab">
    <div v-if="!data" class="sm-loading"><LoaderCircle :size="28" class="spin" /></div>

    <template v-else>
      <section class="achievements-summary" aria-label="Сводка">
        <div class="achievements-summary-count">
          <Trophy :size="28" />
          <p>
            Получено <strong>{{ data.earned_count }}</strong> из {{ data.total_count }}
            <span>{{ plural(data.total_count, ["достижения", "достижений", "достижений"]) }}</span>
          </p>
        </div>
        <div class="timing-track achievements-summary-bar">
          <div class="timing-bar" :style="{ width: `${overall}%` }" />
        </div>
        <p class="achievements-summary-hint">
          Сканы засчитываются, если фото снято кнопкой «Камера» (не из галереи) и вино найдено с уверенностью
          от 70%. Секретные достижения не видны заранее — они появятся здесь, когда вы сделаете что-то подходящее.
        </p>
      </section>

      <section v-if="app.leaderboard.value?.items?.length" class="achievements-leaders" aria-labelledby="leaders-title">
        <div class="achievements-leaders-head">
          <h3 id="leaders-title"><Medal :size="20" /> Топ пользователей</h3>
          <button class="ui-button transparent small" type="button" @click="leaderboardOpen = true">
            Весь рейтинг
            <ChevronRight :size="16" />
          </button>
        </div>
        <ol class="achievements-leaders-list">
          <LeaderRow v-for="entry in app.leaderboard.value.items" :key="entry.rank" :entry="entry" />
          <!-- себя показываем и вне топа -->
          <template v-if="me && !app.leaderboard.value.items.some((entry) => entry.is_me)">
            <li class="leaders-gap" aria-hidden="true">…</li>
            <LeaderRow :entry="me" />
          </template>
        </ol>
      </section>

      <section v-for="group in groups" :key="group.category" class="achievements-group">
        <h3>
          <component :is="categoryIcon(group.category)" :size="20" />
          {{ group.label }}
        </h3>
        <div class="achievements-grid">
          <article v-for="item in group.items" :key="item.code" class="achievement-card" :class="{ earned: item.earned }">
            <span class="achievement-badge">
              <Trophy v-if="item.earned" :size="22" />
              <component :is="categoryIcon(item.category)" v-else :size="22" />
            </span>
            <div class="achievement-card-body">
              <strong>
                {{ item.title }}
                <span v-if="item.secret" class="achievement-secret">секретное</span>
              </strong>
              <span>{{ item.description }}</span>
              <div v-if="!item.earned" class="achievement-progress">
                <div class="timing-track"><div class="timing-bar" :style="{ width: `${ratio(item)}%` }" /></div>
                <small>{{ item.progress }}/{{ item.target }}</small>
              </div>
              <small v-else class="achievement-earned-at">Получено {{ formatDate(item.earned_at) }}</small>
              <small class="achievement-share">{{ formatShare(item.earned_by_percent) }}</small>
            </div>
          </article>
        </div>
      </section>

    </template>

    <LeaderboardModal :open="leaderboardOpen" @close="leaderboardOpen = false" />
  </div>
</template>

<script setup>
import { ChevronRight, LoaderCircle, Medal, Trophy } from "@lucide/vue";

const app = useWineApp();
const leaderboardOpen = ref(false);
const { formatDate } = useWineFormat();
const { categoryIcon, formatShare } = useAchievementFormat();

const data = computed(() => app.achievements.value);
const me = computed(() => app.leaderboard.value?.me);
const overall = computed(() => (data.value?.total_count ? (data.value.earned_count / data.value.total_count) * 100 : 0));

// категории — в порядке каталога; внутри: полученные, затем ближе к цели
const groups = computed(() => {
  const byCategory = new Map();
  for (const item of data.value?.items || []) {
    if (!byCategory.has(item.category)) {
      byCategory.set(item.category, { category: item.category, label: item.category_label, items: [] });
    }
    byCategory.get(item.category).items.push(item);
  }
  for (const group of byCategory.values()) {
    group.items.sort((a, b) => Number(b.earned) - Number(a.earned) || ratio(b) - ratio(a));
  }
  return [...byCategory.values()];
});

function ratio(item) {
  return Math.min(100, (item.progress / item.target) * 100);
}

function plural(count, [one, few, many]) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}

onMounted(() => app.loadAchievements().catch(() => {}));
</script>
