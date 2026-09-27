<template>
  <div class="wine-chat" :class="{ open: terms.chatOpen.value }">
    <section
      v-if="terms.chatOpen.value"
      class="wine-chat-panel"
      role="dialog"
      aria-labelledby="wine-chat-title"
      @keydown.esc="close"
    >
      <header class="wine-chat-head">
        <img src="/icon-192.png" alt="" width="36" height="36" />
        <div>
          <h2 id="wine-chat-title">Винный словарь</h2>
          <small>{{ termsCount }} терминов · нажмите на подчёркнутое слово</small>
        </div>
        <button class="wine-chat-icon-button" type="button" aria-label="Свернуть чат" @click="close">
          <X :size="20" />
        </button>
      </header>

      <div class="wine-chat-search">
        <label>
          <Search :size="18" />
          <input
            ref="searchInput"
            v-model="query"
            type="search"
            placeholder="Найти термин: танин, купаж, терруар…"
            autocomplete="off"
            @keydown.down.prevent="move(1)"
            @keydown.up.prevent="move(-1)"
            @keydown.enter.prevent="choose(results[active])"
          />
        </label>
        <ul v-if="query.trim()" class="wine-chat-results" role="listbox" aria-label="Найденные термины">
          <li v-for="(item, index) in results" :key="item.id">
            <button
              type="button"
              role="option"
              :aria-selected="index === active"
              :class="{ active: index === active }"
              @click="choose(item)"
              @mouseenter="active = index"
            >
              {{ item.term }}
            </button>
          </li>
          <li v-if="!results.length" class="wine-chat-empty">Такого термина нет в словаре</li>
        </ul>
      </div>

      <div ref="feed" class="wine-chat-feed" aria-live="polite">
        <div class="wine-chat-message bot">
          <p>
            Привет! Я подскажу, что значат винные термины. Нажмите на подчёркнутое слово в названии или описании
            вина — или найдите термин через поиск сверху.
          </p>
        </div>

        <article v-for="message in terms.messages.value" :key="message.id" class="wine-chat-message bot">
          <p v-if="message.loading" class="wine-chat-typing"><span /><span /><span /></p>
          <p v-else-if="message.error" class="error-message">{{ message.error }}</p>
          <template v-else>
            <small v-if="message.word && normalize(message.word) !== normalize(message.term.term)" class="wine-chat-word">
              «{{ message.word }}»
            </small>
            <strong>{{ message.term.term }}</strong>
            <p><TermText :text="message.term.definition" :exclude="message.termId" /></p>
            <a v-if="message.term.source_url" :href="message.term.source_url" target="_blank" rel="noreferrer">
              Источник: {{ host(message.term.source_url) }}
            </a>
          </template>
        </article>
      </div>

      <footer v-if="terms.messages.value.length" class="wine-chat-foot">
        <button class="ui-button transparent small" type="button" @click="terms.clearChat()">
          <Eraser :size="16" />
          Очистить
        </button>
      </footer>
    </section>

    <button
      class="wine-chat-launcher"
      type="button"
      :aria-label="terms.chatOpen.value ? 'Свернуть винный словарь' : 'Открыть винный словарь'"
      :aria-expanded="terms.chatOpen.value"
      @click="toggle"
    >
      <img v-if="!terms.chatOpen.value" src="/icon-192.png" alt="" width="60" height="60" />
      <X v-else :size="26" />
    </button>
  </div>
</template>

<script setup>
import { Eraser, Search, X } from "@lucide/vue";

const terms = useWineTerms();
const query = ref("");
const active = ref(0);
const feed = ref(null);
const searchInput = ref(null);

const termsCount = computed(() => terms.index.value?.byId.size || 0);
const results = computed(() => (terms.index.value, terms.searchTerms(query.value)));

function normalize(text) {
  return (text || "").toLowerCase().replaceAll("ё", "е");
}

function host(url) {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

function toggle() {
  terms.chatOpen.value = !terms.chatOpen.value;
}

function close() {
  terms.chatOpen.value = false;
}

function move(delta) {
  if (!results.value.length) return;
  active.value = (active.value + delta + results.value.length) % results.value.length;
}

function choose(item) {
  if (!item) return;
  terms.openTerm(item.id);
  query.value = "";
}

watch(query, () => (active.value = 0));

// новое сообщение — прокручиваем ленту вниз
watch(
  () => terms.messages.value.map((message) => `${message.id}:${message.loading}`).join(),
  async () => {
    await nextTick();
    feed.value?.scrollTo({ top: feed.value.scrollHeight, behavior: "smooth" });
  },
);

// открыли кнопкой (не кликом по слову) — фокус в поиск, но не на телефоне: клавиатура закроет пол-экрана
watch(
  () => terms.chatOpen.value,
  async (open) => {
    if (!open) return;
    terms.loadTerms();
    await nextTick();
    if (!terms.messages.value.length && window.matchMedia("(pointer: fine)").matches) searchInput.value?.focus();
  },
);

onMounted(() => terms.loadTerms());
</script>
