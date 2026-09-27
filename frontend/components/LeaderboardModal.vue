<template>
  <div v-if="open" class="modal-backdrop" @click.self="$emit('close')" @keydown.esc="$emit('close')">
    <section class="leaderboard-modal" role="dialog" aria-modal="true" aria-labelledby="leaderboard-title">
      <button class="close-button" type="button" aria-label="Закрыть" @click="$emit('close')">
        <X :size="20" />
      </button>
      <h2 id="leaderboard-title"><Medal :size="24" /> Рейтинг пользователей</h2>
      <p class="leaderboard-sub">
        {{ data?.total || 0 }} {{ usersWord(data?.total || 0) }} с достижениями
        <template v-if="data?.me">
          · вы на <strong>{{ data.me.rank }}</strong> месте
          <button v-if="!onMyPage" class="link-button" type="button" @click="goToMe">показать</button>
        </template>
      </p>

      <div class="leaderboard-list" :class="{ loading }">
        <ol v-if="data?.items?.length" class="achievements-leaders-list">
          <LeaderRow v-for="entry in data.items" :key="entry.rank" :entry="entry" />
        </ol>
        <p v-else-if="!loading" class="profile-empty">Пока ни у кого нет достижений.</p>
        <p v-if="error" class="error-message">{{ error }}</p>
      </div>

      <nav v-if="pages > 1" class="leaderboard-pages" aria-label="Страницы рейтинга">
        <button class="ui-button transparent small" type="button" :disabled="page <= 1 || loading" @click="load(page - 1)">
          <ChevronLeft :size="18" />
          Назад
        </button>
        <span>{{ page }} из {{ pages }}</span>
        <button class="ui-button transparent small" type="button" :disabled="page >= pages || loading" @click="load(page + 1)">
          Дальше
          <ChevronRight :size="18" />
        </button>
      </nav>
    </section>
  </div>
</template>

<script setup>
import { ChevronLeft, ChevronRight, Medal, X } from "@lucide/vue";

const PAGE_SIZE = 20;

const props = defineProps({
  open: { type: Boolean, default: false },
});
defineEmits(["close"]);

const app = useWineApp();
const data = ref(null);
const page = ref(1);
const loading = ref(false);
const error = ref("");

const pages = computed(() => Math.max(1, Math.ceil((data.value?.total || 0) / PAGE_SIZE)));
const myPage = computed(() => (data.value?.me ? Math.ceil(data.value.me.rank / PAGE_SIZE) : null));
const onMyPage = computed(() => myPage.value === page.value);

async function load(target) {
  loading.value = true;
  error.value = "";
  try {
    data.value = await app.fetchLeaderboard({ limit: PAGE_SIZE, offset: (target - 1) * PAGE_SIZE });
    page.value = target;
  } catch {
    error.value = "Не удалось загрузить рейтинг";
  } finally {
    loading.value = false;
  }
}

function goToMe() {
  if (myPage.value) load(myPage.value);
}

function usersWord(count) {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return "пользователь";
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return "пользователя";
  return "пользователей";
}

// открыли — сразу страница с моим местом
watch(
  () => props.open,
  async (isOpen) => {
    if (!isOpen) return;
    await load(1);
    if (myPage.value && myPage.value !== 1) await load(myPage.value);
  },
);
</script>
