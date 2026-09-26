<template>
  <article
    class="scan-card"
    :class="{ dragging: app.isDragging.value }"
    @dragover.prevent="app.isDragging.value = true"
    @dragleave.prevent="app.isDragging.value = false"
    @drop.prevent="app.handleDrop"
  >
    <div class="scanner-illustration" aria-hidden="true">
      <ScanLine :size="68" />
      <Sparkles :size="24" class="spark spark-one" />
      <Sparkles :size="18" class="spark spark-two" />
    </div>
    <h2>Сканировать этикетку</h2>
    <p>Для телефона откройте камеру. На компьютере перетащите фото сюда или выберите файл.</p>

    <input ref="cameraInput" hidden type="file" accept="image/*" capture="environment" @change="onFileInput" />
    <input ref="galleryInput" hidden type="file" accept="image/*" @change="onFileInput" />

    <button class="ui-button primary scan-action" type="button" @click="cameraInput?.click()">
      <Camera :size="18" />
      Сканировать
    </button>
    <button class="ui-button transparent scan-action" type="button" @click="galleryInput?.click()">
      <Upload :size="18" />
      Загрузить фото
    </button>

    <div v-if="app.selectedFile.value" class="selected-file">
      <img v-if="app.previewUrl.value" :src="app.previewUrl.value" alt="Загруженное фото" />
      <div>
        <strong>{{ app.selectedFile.value.name }}</strong>
        <span>{{ prettyBytes(app.selectedFile.value.size) }}</span>
      </div>
    </div>

    <button
      class="ui-button primary large"
      type="button"
      :disabled="!app.selectedFile.value || app.isSearching.value"
      @click="app.searchWine"
    >
      <LoaderCircle v-if="app.isSearching.value" :size="18" class="spin" />
      <Search v-else :size="18" />
      {{ app.isSearching.value ? "Ищем вино" : "Найти вино" }}
    </button>

    <p v-if="app.errorMessage.value" class="error-message">{{ app.errorMessage.value }}</p>
  </article>
</template>

<script setup>
import { Camera, LoaderCircle, ScanLine, Search, Sparkles, Upload } from "@lucide/vue";

const app = useWineApp();
const { prettyBytes } = useWineFormat();

const cameraInput = ref(null);
const galleryInput = ref(null);

function onFileInput(event) {
  app.setSelectedFile(event.target.files?.[0]);
  event.target.value = "";
}
</script>
