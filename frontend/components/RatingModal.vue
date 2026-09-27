<template>
  <div v-if="dialog" class="modal-backdrop rating-backdrop" @click.self="close">
    <form class="rating-modal" @submit.prevent="save">
      <button class="close-button" type="button" aria-label="Закрыть" @click="close">
        <X :size="20" />
      </button>

      <div class="rating-wine">
        <div class="viewed-wine-photo rating-wine-photo">
          <img v-if="mainPhoto(dialog.wine)" :src="mainPhoto(dialog.wine)" :alt="dialog.wine?.name" />
        </div>
        <div>
          <strong>{{ dialog.wine?.name }}</strong>
          <span>{{ dialog.wine?.producer }}</span>
        </div>
      </div>

      <h2>Поставь свою оценку</h2>

      <div class="glass-rating" role="radiogroup" aria-label="Оценка от 1 до 5" @mouseleave="hover = 0">
        <button
          v-for="value in 5"
          :key="value"
          type="button"
          role="radio"
          :aria-checked="dialog.rating === value"
          :aria-label="`${value} из 5 — ${LABELS[value]}`"
          :class="{ active: value <= shown }"
          @mouseenter="hover = value"
          @focus="hover = value"
          @blur="hover = 0"
          @click="dialog.rating = dialog.rating === value ? null : value"
        >
          <WineGlass :filled="value <= shown" :size="40" />
        </button>
      </div>
      <p class="glass-rating-caption">{{ shown ? LABELS[shown] : "Выберите от 1 до 5 бокалов" }}</p>

      <label class="rating-comment">
        <span>Комментарий <small>— по желанию</small></span>
        <textarea
          v-model="dialog.comment"
          rows="4"
          maxlength="2000"
          placeholder="Чем запомнилось вино: вкус, к чему подошло, возьмёте ли ещё"
        />
        <small class="rating-counter">{{ (dialog.comment || "").length }} / 2000</small>
      </label>

      <div class="rating-photos">
        <span>Фото <small>— до {{ MAX_PHOTOS }}, по желанию</small></span>
        <div class="rating-photo-grid">
          <figure v-for="photo in dialog.photos" :key="photo.id" class="rating-photo">
            <img :src="photo.url" alt="" />
            <button type="button" aria-label="Убрать фото" :disabled="saving" @click="removeSaved(photo)">
              <X :size="14" />
            </button>
          </figure>
          <figure v-for="(item, index) in dialog.newPhotos" :key="item.preview" class="rating-photo">
            <img :src="item.preview" alt="" />
            <button type="button" aria-label="Убрать фото" :disabled="saving" @click="removeNew(index)">
              <X :size="14" />
            </button>
          </figure>
          <label v-if="photoCount < MAX_PHOTOS" class="rating-photo-add" :class="{ disabled: saving }">
            <ImagePlus :size="22" />
            <span>Добавить</span>
            <input class="visually-hidden" type="file" accept="image/*" multiple :disabled="saving" @change="addPhotos" />
          </label>
        </div>
      </div>

      <p v-if="error" class="error-message">{{ error }}</p>

      <div class="rating-modal-actions">
        <button class="ui-button primary large" type="submit" :disabled="!canSave || saving">
          <LoaderCircle v-if="saving" :size="18" class="spin" />
          {{ dialog.existing ? "Сохранить изменения" : "Оценить" }}
        </button>
        <button v-if="dialog.existing" class="ui-button tertiary" type="button" :disabled="saving" @click="remove">
          Удалить отзыв
        </button>
        <button v-else class="ui-button transparent" type="button" @click="close">Отмена</button>
      </div>
    </form>
  </div>
</template>

<script setup>
import { ImagePlus, LoaderCircle, X } from "@lucide/vue";

const MAX_PHOTOS = 5;
const LABELS = { 1: "Не понравилось", 2: "Так себе", 3: "Неплохо", 4: "Хорошее вино", 5: "Отлично" };

const app = useWineApp();
const { mainPhoto } = useWineFormat();
const dialog = app.ratingDialog;
const hover = ref(0);
const saving = ref(false);
const error = ref("");

const shown = computed(() => hover.value || dialog.value?.rating || 0);
const canSave = computed(() => Boolean(dialog.value?.rating || dialog.value?.comment?.trim()));
const photoCount = computed(() => (dialog.value?.photos.length || 0) + (dialog.value?.newPhotos.length || 0));

function addPhotos(event) {
  const files = [...(event.target.files || [])].filter((file) => file.type.startsWith("image/"));
  event.target.value = "";
  for (const file of files.slice(0, MAX_PHOTOS - photoCount.value)) {
    dialog.value.newPhotos.push({ file, preview: URL.createObjectURL(file) });
  }
}

function removeNew(index) {
  const [item] = dialog.value.newPhotos.splice(index, 1);
  if (item) URL.revokeObjectURL(item.preview);
}

function removeSaved(photo) {
  dialog.value.photos = dialog.value.photos.filter((item) => item.id !== photo.id);
  dialog.value.removedPhotoIds.push(photo.id);
}

// превью живут, пока открыт диалог
watch(dialog, (_, previous) => {
  previous?.newPhotos?.forEach((item) => URL.revokeObjectURL(item.preview));
  hover.value = 0;
  error.value = "";
});

function close() {
  if (!saving.value) dialog.value = null;
}

async function save() {
  if (!canSave.value) return;
  saving.value = true;
  error.value = "";
  try {
    await app.saveReview();
  } catch (exc) {
    error.value = exc?.data?.detail || "Не удалось сохранить оценку";
  } finally {
    saving.value = false;
  }
}

async function remove() {
  saving.value = true;
  error.value = "";
  try {
    await app.deleteReview();
  } catch (exc) {
    error.value = exc?.data?.detail || "Не удалось удалить отзыв";
  } finally {
    saving.value = false;
  }
}
</script>
