export function useWineApp() {
  const activeView = useState("wine.activeView", () => "scanner");
  const profileTab = useState("wine.profileTab", () => "history");
  const authDialog = useState("wine.authDialog", () => false);
  const authMode = useState("wine.authMode", () => "login");
  const authError = useState("wine.authError", () => "");
  const authForm = useState("wine.authForm", () => ({ email: "", password: "" }));
  const token = useState("wine.token", () => "");
  const user = useState("wine.user", () => null);

  const selectedFile = useState("wine.selectedFile", () => null);
  const previewUrl = useState("wine.previewUrl", () => "");
  const isDragging = useState("wine.isDragging", () => false);
  const isSearching = useState("wine.isSearching", () => false);
  const errorMessage = useState("wine.errorMessage", () => "");
  const searchResponse = useState("wine.searchResponse", () => null);
  // фото, по которому выполнен текущий поиск: на нём рисуем bbox в диагностике
  const searchedImageUrl = useState("wine.searchedImageUrl", () => "");
  // исходное фото и его детекции после поиска по всему фото: { url, width, height, labels, bottles }
  const photoSource = useState("wine.photoSource", () => null);
  // выбор пользователя на исходном фото: { box, kind: "label" | "manual" }; null — автовыбор бэкенда
  const pick = useState("wine.pick", () => null);
  const selectedPhotos = useState("wine.selectedPhotos", () => ({}));
  const showGenerated = useState("wine.showGenerated", () => false);
  const showDiagnostics = useState("wine.showDiagnostics", () => false);

  const history = useState("wine.history", () => ({ items: [] }));
  const favorites = useState("wine.favorites", () => ({ items: [] }));
  const reviews = useState("wine.reviews", () => ({ items: [] }));
  const notifications = useState("wine.notifications", () => ({ items: [], unread: 0 }));
  const notificationDetail = useState("wine.notificationDetail", () => null);
  // окно оценки: { wine, wineId, notificationId, rating, comment, existing }
  const ratingDialog = useState("wine.ratingDialog", () => null);
  // растёт после каждого изменения своего отзыва — блоки отзывов перезагружаются
  const reviewsVersion = useState("wine.reviewsVersion", () => 0);
  // окно с информацией о вине (из истории, избранного, отзывов): { wine, searchId, context }
  const wineDetail = useState("wine.wineDetail", () => null);
  const initialized = useState("wine.initialized", () => false);

  const bestMatch = computed(() => searchResponse.value?.results?.[0] || null);
  const similarMatches = computed(() =>
    (searchResponse.value?.results || []).filter((item) => item.wine_id !== bestMatch.value?.wine_id).slice(0, 12),
  );
  const profileTabs = computed(() => [
    { id: "history", label: "История", count: history.value.items.length },
    { id: "favorites", label: "Избранное", count: favorites.value.items.length },
    { id: "reviews", label: "Отзывы", count: reviews.value.items.length },
    { id: "notifications", label: "Уведомления", count: notifications.value.items.length },
  ]);

  function authHeaders() {
    return token.value ? { Authorization: `Bearer ${token.value}` } : {};
  }

  async function apiFetch(path, options = {}) {
    const headers = { ...(options.headers || {}), ...authHeaders() };
    return await $fetch(path, { ...options, headers, credentials: "include" });
  }

  async function initialize() {
    if (initialized.value) return;
    initialized.value = true;
    token.value = localStorage.getItem("wine_token") || "";
    if (token.value) {
      await loadMe();
    }
    await loadPublicHistory();
    if (user.value) {
      await refreshPrivateData();
    }
  }

  async function loadMe() {
    try {
      user.value = await apiFetch("/api/v1/auth/me");
    } catch {
      token.value = "";
      localStorage.removeItem("wine_token");
      user.value = null;
    }
  }

  async function submitAuth() {
    authError.value = "";
    try {
      const response = await $fetch(`/api/v1/auth/${authMode.value}`, {
        method: "POST",
        body: { email: authForm.value.email, password: authForm.value.password },
        credentials: "include",
      });
      token.value = response.access_token;
      localStorage.setItem("wine_token", token.value);
      user.value = response.user;
      authDialog.value = false;
      await refreshPrivateData();
    } catch (error) {
      authError.value = error?.data?.detail || "Не удалось выполнить вход";
    }
  }

  async function refreshPrivateData() {
    await Promise.allSettled([loadPublicHistory(), loadFavorites(), loadReviews(), loadNotifications()]);
  }

  async function loadPublicHistory() {
    try {
      const response = await apiFetch("/api/v1/history?limit=30");
      history.value = { items: response.items || [] };
    } catch {
      history.value = { items: [] };
    }
  }

  async function loadFavorites() {
    if (!user.value) return;
    const response = await apiFetch("/api/v1/favorites");
    favorites.value = { items: response.items || [] };
  }

  async function loadReviews() {
    if (!user.value) return;
    const response = await apiFetch("/api/v1/reviews");
    reviews.value = { items: response.items || [] };
  }

  async function loadNotifications() {
    if (!user.value) return;
    const response = await apiFetch("/api/v1/notifications");
    notifications.value = { items: response.items || [], unread: response.unread || 0 };
  }

  function setSelectedFile(file) {
    if (!file) return;
    if (!file.type?.startsWith("image/")) {
      errorMessage.value = "Выберите изображение";
      return;
    }
    selectedFile.value = file;
    errorMessage.value = "";
    const previous = previewUrl.value;
    previewUrl.value = URL.createObjectURL(file);
    revokeIfUnused(previous);
  }

  function handleDrop(event) {
    isDragging.value = false;
    setSelectedFile(event.dataTransfer.files?.[0]);
  }

  // object URL живёт, пока его показывает превью, исходное фото выбора или диагностика
  function revokeIfUnused(url) {
    if (url && ![previewUrl.value, photoSource.value?.url, searchedImageUrl.value].includes(url)) {
      URL.revokeObjectURL(url);
    }
  }

  async function runSearch(file) {
    isSearching.value = true;
    errorMessage.value = "";
    showGenerated.value = false;
    try {
      const form = new FormData();
      form.append("image", file);
      searchResponse.value = await apiFetch("/api/v1/search/image/extended?limit=12&debug=true", {
        method: "POST",
        body: form,
      });
      if (!searchResponse.value?.results?.length) {
        errorMessage.value = "Вино не найдено. Попробуйте фото этикетки крупнее и без бликов.";
      }
      await loadPublicHistory();
      return true;
    } catch (error) {
      errorMessage.value = error?.data?.detail || "Не удалось выполнить поиск";
      return false;
    } finally {
      isSearching.value = false;
    }
  }

  function setSearchedImage(url) {
    const previous = searchedImageUrl.value;
    searchedImageUrl.value = url;
    revokeIfUnused(previous);
  }

  // поиск по всему фото: бэкенд сам выбирает бутылку и этикетку («автовыбор»)
  async function searchWine() {
    if (!selectedFile.value) return;
    const url = previewUrl.value;
    if (!(await runSearch(selectedFile.value))) return;
    const response = searchResponse.value;
    const previousSource = photoSource.value?.url;
    photoSource.value = response?.image
      ? {
          url,
          width: response.image.width,
          height: response.image.height,
          labels: response.detections?.labels || [],
          bottles: response.detections?.bottles || [],
        }
      : null;
    pick.value = null;
    setSearchedImage(url);
    revokeIfUnused(previousSource);
  }

  // вырезаем область исходного фото в браузере и ищем по ней как по обычной картинке
  async function cropToFile(source, [x1, y1, x2, y2]) {
    // onload, а не image.decode(): decode не завершается, пока вкладка скрыта
    const image = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("image load failed"));
      img.src = source.url;
    });
    // детекции в пикселях бэкенда (фото после EXIF-поворота); браузер рисует так же, но масштаб сверяем
    const scale = image.naturalWidth / source.width;
    const width = Math.max(1, Math.round((x2 - x1) * scale));
    const height = Math.max(1, Math.round((y2 - y1) * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(image, x1 * scale, y1 * scale, width, height, 0, 0, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    return new File([blob], "area.jpg", { type: "image/jpeg" });
  }

  async function searchArea(box, kind, extra = {}) {
    const source = photoSource.value;
    if (!source || isSearching.value) return;
    // занимаем «поиск» уже на время вырезки, иначе быстрые повторные клики запускают гонку запросов
    isSearching.value = true;
    let file;
    try {
      file = await cropToFile(source, box);
    } catch {
      errorMessage.value = "Не удалось вырезать область фото";
      isSearching.value = false;
      return;
    }
    const url = URL.createObjectURL(file);
    if (!(await runSearch(file))) {
      URL.revokeObjectURL(url);
      return;
    }
    pick.value = { box, kind, ...extra };
    setSearchedImage(url);
  }

  async function addFavorite(match, searchId = searchResponse.value?.search_id) {
    if (!user.value) {
      authDialog.value = true;
      return;
    }
    await apiFetch("/api/v1/favorites", {
      method: "POST",
      body: { wine_id: match.wine_id || match.wine?.id, search_id: searchId || null },
    });
    await loadFavorites();
    if (notificationDetail.value) await openNotificationDetail(notificationDetail.value.id);
  }

  async function removeFavorite(wineId) {
    await apiFetch(`/api/v1/favorites/${wineId}`, { method: "DELETE" });
    await loadFavorites();
  }

  function openRating({ wine, wineId = wine?.id, notificationId = null, review = null, rating = null }) {
    if (!user.value) {
      authDialog.value = true;
      return;
    }
    const existing = review || reviews.value.items.find((item) => item.wine_id === wineId) || null;
    ratingDialog.value = {
      wine,
      wineId,
      notificationId,
      rating: rating ?? existing?.rating ?? null,
      comment: existing?.comment || "",
      existing: Boolean(existing),
    };
  }

  async function afterReviewChange() {
    ratingDialog.value = null;
    reviewsVersion.value += 1;
    await loadReviews();
    if (notificationDetail.value) await openNotificationDetail(notificationDetail.value.id);
  }

  async function saveReview() {
    const dialog = ratingDialog.value;
    if (!dialog) return;
    await apiFetch(`/api/v1/reviews/${dialog.wineId}`, {
      method: "PUT",
      body: { rating: dialog.rating || null, comment: dialog.comment?.trim() || null, notification_id: dialog.notificationId },
    });
    await afterReviewChange();
  }

  async function deleteReview() {
    const dialog = ratingDialog.value;
    if (!dialog) return;
    await apiFetch(`/api/v1/reviews/${dialog.wineId}`, { method: "DELETE" });
    await afterReviewChange();
  }

  async function fetchWineReviews(wineId) {
    return await apiFetch(`/api/v1/wines/${wineId}/reviews`);
  }

  function openWine(wine, { searchId = null, context = "" } = {}) {
    if (!wine) return;
    wineDetail.value = { wine, searchId, context };
  }

  async function markAllNotificationsRead() {
    await apiFetch("/api/v1/notifications/read-all", { method: "POST" });
    await loadNotifications();
  }

  async function openNotifications() {
    if (!user.value) {
      authDialog.value = true;
      return;
    }
    activeView.value = "profile";
    profileTab.value = "notifications";
    await loadNotifications();
  }

  async function openNotificationDetail(id) {
    notificationDetail.value = await apiFetch(`/api/v1/notifications/${id}`);
    await loadNotifications();
  }

  return {
    activeView,
    addFavorite,
    authDialog,
    authError,
    authForm,
    authMode,
    bestMatch,
    errorMessage,
    favorites,
    handleDrop,
    history,
    initialize,
    isDragging,
    isSearching,
    loadNotifications,
    notificationDetail,
    notifications,
    openNotificationDetail,
    openNotifications,
    previewUrl,
    profileTab,
    profileTabs,
    refreshPrivateData,
    removeFavorite,
    reviews,
    searchResponse,
    searchWine,
    searchedImageUrl,
    photoSource,
    pick,
    searchArea,
    selectedFile,
    selectedPhotos,
    openRating,
    openWine,
    wineDetail,
    reviewsVersion,
    fetchWineReviews,
    ratingDialog,
    saveReview,
    deleteReview,
    markAllNotificationsRead,
    setSelectedFile,
    showDiagnostics,
    showGenerated,
    similarMatches,
    submitAuth,
    user,
  };
}
