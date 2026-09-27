<template>
  <span class="term-text">
    <template v-for="(part, index) in parts" :key="index">
      <span
        v-if="part.termId"
        class="term-link"
        role="button"
        tabindex="0"
        :title="`Что такое «${termName(part.termId)}»?`"
        @click.stop.prevent="terms.openTerm(part.termId, part.text)"
        @keydown.enter.stop.prevent="terms.openTerm(part.termId, part.text)"
      >{{ part.text }}</span>
      <template v-else>{{ part.text }}</template>
    </template>
  </span>
</template>

<script setup>
// текст с подсвеченными винными терминами; клик открывает справку в чате
const props = defineProps({
  text: { type: String, default: "" },
  exclude: { type: String, default: "" }, // термин, который не подсвечиваем (в его же определении)
});

const terms = useWineTerms();
const parts = computed(() =>
  (terms.index.value, terms.segments(props.text)).map((part) =>
    part.termId && part.termId === props.exclude ? { text: part.text } : part,
  ),
);

function termName(termId) {
  return terms.index.value?.byId.get(termId)?.term || "";
}

onMounted(() => terms.loadTerms());
</script>
