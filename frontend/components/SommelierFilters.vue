<template>
  <div class="sm-filters-wrap" :class="{ open }" @click.self="$emit('close')">
    <aside class="sm-filters" aria-label="Фильтры">
      <div class="sm-filters-mobile-head">
        <h2>Фильтры</h2>
        <button class="close-button" type="button" aria-label="Закрыть фильтры" @click="$emit('close')">
          <X :size="20" />
        </button>
      </div>

      <section v-for="group in groups" :key="group.key" class="sm-filter-group">
        <button class="sm-filter-title" type="button" :aria-expanded="!collapsed[group.key]" @click="collapsed[group.key] = !collapsed[group.key]">
          {{ group.title }}
          <ChevronDown :size="20" :class="{ flipped: !collapsed[group.key] }" />
        </button>

        <div v-show="!collapsed[group.key]" class="sm-filter-body">
          <label v-if="group.searchable" class="sm-filter-search">
            <input v-model="search[group.key]" type="search" placeholder="Поиск" />
            <Search :size="18" />
          </label>

          <label
            v-for="option in visible(group)"
            :key="option.value"
            class="sm-check"
            :class="{ empty: !option.count && !selected(group.key, option.value) }"
          >
            <input
              type="checkbox"
              :checked="selected(group.key, option.value)"
              :disabled="!option.count && !selected(group.key, option.value)"
              @change="$emit('toggle', group.key, option.value)"
            />
            <span>{{ capitalize(option.label) }}</span>
            <small>{{ option.count }}</small>
          </label>

          <button
            v-if="group.limit && matching(group).length > group.limit && !search[group.key]"
            class="link-button"
            type="button"
            @click="expanded[group.key] = !expanded[group.key]"
          >
            {{ expanded[group.key] ? "Свернуть" : group.more }}
          </button>
        </div>
      </section>

      <section class="sm-filter-group">
        <button class="sm-filter-title" type="button" :aria-expanded="!collapsed.rating" @click="collapsed.rating = !collapsed.rating">
          Народный рейтинг
          <ChevronDown :size="20" :class="{ flipped: !collapsed.rating }" />
        </button>
        <div v-show="!collapsed.rating" class="sm-chips">
          <button
            v-for="option in RATINGS"
            :key="option.label"
            type="button"
            class="sm-chip small"
            :class="{ selected: filters.min_rating === option.value }"
            @click="$emit('rating', option.value)"
          >
            {{ option.label }}
          </button>
        </div>
      </section>

      <div class="sm-filters-footer">
        <button class="ui-button transparent" type="button" @click="$emit('reset')">Сбросить фильтры</button>
        <button class="ui-button primary sm-filters-apply" type="button" @click="$emit('close')">
          Показать {{ total }}
        </button>
      </div>
    </aside>
  </div>
</template>

<script setup>
import { ChevronDown, Search, X } from "@lucide/vue";

const props = defineProps({
  facets: { type: Object, default: null },
  filters: { type: Object, required: true },
  open: { type: Boolean, default: false },
  total: { type: Number, default: 0 },
});

defineEmits(["toggle", "rating", "close", "reset"]);

const RATINGS = [
  { label: "Любой", value: null },
  { label: "5", value: 5 },
  { label: "4.5+", value: 4.5 },
  { label: "4+", value: 4 },
];

const groups = [
  { key: "type", facet: "types", title: "Тип" },
  { key: "color", facet: "colors", title: "Цвет" },
  { key: "sugar", facet: "sugars", title: "Сахар" },
  { key: "region", facet: "regions", title: "Регион", limit: 6, more: "Все регионы" },
  { key: "grape", facet: "grapes", title: "Сорт винограда", searchable: true, limit: 6, more: "Все сорта" },
];

const collapsed = reactive({});
const expanded = reactive({});
const search = reactive({});

function selected(key, value) {
  return props.filters[key].includes(value);
}

function matching(group) {
  const needle = (search[group.key] || "").trim().toLowerCase();
  const options = props.facets?.[group.facet] || [];
  if (!needle) return options;
  return options.filter((option) => option.label.toLowerCase().includes(needle));
}

// выбранные — всегда наверху, чтобы их было видно и в свёрнутом списке
function visible(group) {
  const options = matching(group);
  if (!group.limit || expanded[group.key] || search[group.key]) return options;
  const chosen = options.filter((option) => selected(group.key, option.value));
  const rest = options.filter((option) => !selected(group.key, option.value) && option.count);
  return [...chosen, ...rest].slice(0, Math.max(group.limit, chosen.length));
}

function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
</script>
