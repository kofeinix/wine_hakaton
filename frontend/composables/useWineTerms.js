// Словарь винных терминов и чат-справочник.
// Правило совпадения — с бэкенда (GET /api/v1/terms): слово = основа термина + одно из окончаний.
// Чат живёт только в памяти страницы: не сохраняется ни в аккаунте, ни в браузере.

const WORD = /[a-zа-я0-9]+(?:-[a-zа-я0-9]+)*/g;
const segmentCache = new Map();
const detailCache = new Map();
let indexRequest = null;
let messageSeq = 0;

function normalize(text) {
  return text.toLowerCase().replaceAll("ё", "е");
}

export function useWineTerms() {
  const index = useState("terms.index", () => null); // { endings: Set, items: [...] } по убыванию длины фразы
  const chatOpen = useState("terms.chatOpen", () => false);
  const messages = useState("terms.messages", () => []);

  async function loadTerms() {
    if (index.value) return index.value;
    indexRequest ||= $fetch("/api/v1/terms")
      .then((response) => {
        index.value = {
          endings: new Set(response.endings),
          // фразы проверяем раньше отдельных слов: «Игристое вино», а не только «вино»
          items: [...response.items].sort((a, b) => b.stems.length - a.stems.length),
          byId: new Map(response.items.map((item) => [item.id, item])),
        };
        segmentCache.clear();
        return index.value;
      })
      .catch(() => {
        indexRequest = null; // словарь не обязателен — попробуем в следующий раз
        return null;
      });
    return indexRequest;
  }

  function wordMatches(word, stem, exact) {
    if (!word.startsWith(stem)) return false;
    const ending = word.slice(stem.length);
    return exact ? ending === "" : index.value.endings.has(ending);
  }

  // текст → [{ text, termId? }]; каждый термин подсвечивается один раз — первое упоминание
  function segments(text) {
    if (!text) return [];
    if (!index.value) return [{ text }];
    if (segmentCache.has(text)) return segmentCache.get(text);
    const normalized = normalize(text); // та же длина, позиции совпадают с исходным текстом
    const words = [...normalized.matchAll(WORD)].map((match) => ({
      word: match[0],
      start: match.index,
      end: match.index + match[0].length,
    }));
    const found = [];
    const used = new Set();
    for (let i = 0; i < words.length; i += 1) {
      for (const term of index.value.items) {
        const count = term.stems.length;
        if (used.has(term.id) || i + count > words.length) continue;
        let ok = true;
        for (let k = 0; k < count && ok; k += 1) {
          ok = wordMatches(words[i + k].word, term.stems[k], term.exact);
          // слова фразы — рядом, только через пробелы
          if (ok && k > 0 && normalized.slice(words[i + k - 1].end, words[i + k].start).trim()) ok = false;
        }
        if (!ok) continue;
        found.push({ start: words[i].start, end: words[i + count - 1].end, termId: term.id });
        used.add(term.id);
        i += count - 1;
        break;
      }
    }
    const result = [];
    let cursor = 0;
    for (const match of found) {
      if (match.start > cursor) result.push({ text: text.slice(cursor, match.start) });
      result.push({ text: text.slice(match.start, match.end), termId: match.termId });
      cursor = match.end;
    }
    if (cursor < text.length) result.push({ text: text.slice(cursor) });
    segmentCache.set(text, result);
    return result;
  }

  function searchTerms(query, limit = 8) {
    const needle = normalize(query.trim());
    if (!needle || !index.value) return [];
    const items = [...index.value.byId.values()];
    const starts = items.filter((item) => normalize(item.term).startsWith(needle));
    const contains = items.filter((item) => !starts.includes(item) && normalize(item.term).includes(needle));
    return [...starts, ...contains].slice(0, limit);
  }

  async function fetchTerm(termId) {
    if (!detailCache.has(termId)) {
      detailCache.set(
        termId,
        $fetch(`/api/v1/terms/${termId}`).catch((error) => {
          detailCache.delete(termId);
          throw error;
        }),
      );
    }
    return detailCache.get(termId);
  }

  // клик по слову или выбор в поиске: открыть чат и добавить справку новым сообщением
  async function openTerm(termId, word = "") {
    chatOpen.value = true;
    const id = ++messageSeq;
    messages.value = [...messages.value, { id, kind: "term", termId, word, loading: true }];
    let patch;
    try {
      const term = await fetchTerm(termId);
      patch = { loading: false, term };
    } catch {
      patch = { loading: false, error: "Не удалось загрузить справку. Попробуйте ещё раз." };
    }
    messages.value = messages.value.map((message) => (message.id === id ? { ...message, ...patch } : message));
  }

  function clearChat() {
    messages.value = [];
  }

  return { chatOpen, clearChat, index, loadTerms, messages, openTerm, searchTerms, segments };
}
