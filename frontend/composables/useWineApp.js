const CAMERA_FRESH_MS = 2 * 60 * 1000;

export function useWineApp() {
  const activeView = useState("wine.activeView", () => "scanner");
  const profileTab = useState("wine.profileTab", () => "history");
  const authDialog = useState("wine.authDialog", () => false);
  const authMode = useState("wine.authMode", () => "login");
  const authError = useState("wine.authError", () => "");
  const authForm = useState("wine.authForm", () => ({ email: "", password: "" }));
  // действие гостя, прерванное входом (оценить, в избранное) — выполняем сразу после входа
  const pendingAction = useState("wine.pendingAction", () => null);
  const token = useState("wine.token", () => "");
  const user = useState("wine.user", () => null);

  const selectedFile = useState("wine.selectedFile", () => null);
  // фото только что снято кнопкой «Камера» — только такие сканы идут в достижения
  const selectedFromCamera = useState("wine.selectedFromCamera", () => false);
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
  const achievements = useState("wine.achievements", () => null);
  const leaderboard = useState("wine.leaderboard", () => null);
  const achievementToasts = useState("wine.achievementToasts", () => []);

  const bestMatch = computed(() => searchResponse.value?.results?.[0] || null);
  const similarMatches = computed(() =>
    (searchResponse.value?.results || []).filter((item) => item.wine_id !== bestMatch.value?.wine_id).slice(0, 12),
  );
  const profileTabs = computed(() => [
    { id: "history", label: "История", count: history.value.items.length },
    { id: "favorites", label: "Избранное", count: favorites.value.items.length },
    { id: "reviews", label: "Отзывы", count: reviews.value.items.length },
    { id: "notifications", label: "Уведомления", count: notifications.value.items.length },
    { id: "achievements", label: "Достижения", count: achievements.value?.earned_count || 0 },
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
      const action = pendingAction.value;
      pendingAction.value = null;
      if (action) await action();
    } catch (error) {
      authError.value = error?.data?.detail || "Не удалось выполнить вход";
    }
  }

  async function refreshPrivateData() {
    await Promise.allSettled([loadPublicHistory(), loadFavorites(), loadReviews(), loadNotifications()]);
    // при входе подхватываем то, что получено без нас (лайки под комментарием) и за перенесённую историю
    await checkAchievements();
    await loadAchievements().catch(() => {});
  }

  // --- достижения ---------------------------------------------------------------------

  let toastSeq = 0;

  // прогресс считается на сервере по данным пользователя; вызываем после каждого действия
  async function checkAchievements() {
    if (!user.value) return;
    try {
      const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const response = await apiFetch("/api/v1/achievements/check", { method: "POST", body: { timezone } });
      const updates = response.updates || [];
      const earned = updates.filter((update) => update.type === "earned");
      // несколько полученных за одно действие — одним уведомлением, а не лентой
      const toasts =
        earned.length > 1
          ? [
              {
                type: "earned",
                code: "several",
                category: "meta",
                title: `Получено ${earned.length} ${plural(earned.length, ["достижение", "достижения", "достижений"])}`,
                description: earned.map((update) => `«${update.title}»`).join(", "),
                progress: earned.length,
                target: earned.length,
              },
              ...updates.filter((update) => update.type !== "earned"),
            ]
          : updates;
      achievementToasts.value = [...achievementToasts.value, ...toasts.map((toast) => ({ ...toast, id: ++toastSeq }))];
      if (response.updates?.length && (achievements.value || profileTab.value === "achievements")) {
        await loadAchievements();
      }
      // полученные достижения пишутся и в уведомления
      if (updates.some((update) => update.type !== "progress")) await loadNotifications();
    } catch {
      // достижения не должны ломать основное действие
    }
  }

  function plural(count, [one, few, many]) {
    const mod10 = count % 10;
    const mod100 = count % 100;
    if (mod10 === 1 && mod100 !== 11) return one;
    if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
    return many;
  }

  function dismissToast(id) {
    achievementToasts.value = achievementToasts.value.filter((toast) => toast.id !== id);
  }

  async function loadAchievements() {
    if (!user.value) return;
    const [list, top] = await Promise.all([
      apiFetch("/api/v1/achievements"),
      apiFetch("/api/v1/achievements/leaderboard"),
    ]);
    achievements.value = list;
    leaderboard.value = top;
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

  // camera — выбрано кнопкой «Камера». На компьютере она открывает обычный выбор файла, поэтому
  // дополнительно требуем свежий файл: у снимка с камеры lastModified ≈ сейчас, у старого фото — нет
  function setSelectedFile(file, { camera = false } = {}) {
    if (!file) return;
    if (!file.type?.startsWith("image/")) {
      errorMessage.value = "Выберите изображение";
      return;
    }
    selectedFile.value = file;
    selectedFromCamera.value = camera && Date.now() - file.lastModified < CAMERA_FRESH_MS;
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
      const source = selectedFromCamera.value ? "camera" : "file";
      searchResponse.value = await apiFetch(`/api/v1/search/image/extended?limit=12&debug=true&source=${source}`, {
        method: "POST",
        body: form,
      });
      if (!searchResponse.value?.results?.length) {
        errorMessage.value = "Вино не найдено. Попробуйте фото этикетки крупнее и без бликов.";
      }
      await loadPublicHistory();
      checkAchievements();
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

  function requireAuth(action) {
    pendingAction.value = action;
    authDialog.value = true;
  }

  async function addFavorite(match, searchId = searchResponse.value?.search_id) {
    if (!user.value) {
      requireAuth(() => addFavorite(match, searchId));
      return;
    }
    await apiFetch("/api/v1/favorites", {
      method: "POST",
      body: { wine_id: match.wine_id || match.wine?.id, search_id: searchId || null },
    });
    await loadFavorites();
    checkAchievements();
    if (notificationDetail.value) await openNotificationDetail(notificationDetail.value.id);
  }

  async function removeFavorite(wineId) {
    await apiFetch(`/api/v1/favorites/${wineId}`, { method: "DELETE" });
    await loadFavorites();
  }

  function openRating({ wine, wineId = wine?.id, notificationId = null, review = null, rating = null }) {
    if (!user.value) {
      requireAuth(() => openRating({ wine, wineId, notificationId, review, rating }));
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
      photos: [...(existing?.photos || [])],
      newPhotos: [],
      removedPhotoIds: [],
    };
  }

  async function afterReviewChange() {
    ratingDialog.value = null;
    reviewsVersion.value += 1;
    await loadReviews();
    checkAchievements();
    if (notificationDetail.value) await openNotificationDetail(notificationDetail.value.id);
  }

  async function saveReview() {
    const dialog = ratingDialog.value;
    if (!dialog) return;
    await apiFetch(`/api/v1/reviews/${dialog.wineId}`, {
      method: "PUT",
      body: { rating: dialog.rating || null, comment: dialog.comment?.trim() || null, notification_id: dialog.notificationId },
    });
    // фото — после отзыва (без отзыва их некуда прикрепить); выполненное сразу убираем из очереди,
    // чтобы при ошибке повторное «Сохранить» не повторяло его
    while (dialog.removedPhotoIds.length) {
      await apiFetch(`/api/v1/reviews/${dialog.wineId}/photos/${dialog.removedPhotoIds[0]}`, { method: "DELETE" });
      dialog.removedPhotoIds.shift();
    }
    while (dialog.newPhotos.length) {
      const form = new FormData();
      form.append("image", dialog.newPhotos[0].file);
      dialog.photos.push(await apiFetch(`/api/v1/reviews/${dialog.wineId}/photos`, { method: "POST", body: form }));
      URL.revokeObjectURL(dialog.newPhotos.shift().preview);
    }
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

  async function reactToReview(wineId, reviewUserId, value) {
    if (!user.value) {
      // после входа отзыв может оказаться своим — на него реакцию не ставят (404), это не ошибка входа
      requireAuth(() => reactToReview(wineId, reviewUserId, value).catch(() => {}));
      return;
    }
    await apiFetch(`/api/v1/wines/${wineId}/reviews/${reviewUserId}/reaction`, {
      method: "PUT",
      body: { value },
    });
    reviewsVersion.value += 1;
    checkAchievements();
  }

  async function clearReviewReaction(wineId, reviewUserId) {
    if (!user.value) return;
    await apiFetch(`/api/v1/wines/${wineId}/reviews/${reviewUserId}/reaction`, { method: "DELETE" });
    reviewsVersion.value += 1;
  }

  async function updateProfileSettings(settings) {
    if (!user.value) return;
    user.value = await apiFetch("/api/v1/profile", { method: "PATCH", body: settings });
  }

  async function uploadAvatar(file) {
    if (!user.value || !file) return;
    const form = new FormData();
    form.append("image", file);
    user.value = await apiFetch("/api/v1/profile/avatar", { method: "POST", body: form });
    reviewsVersion.value += 1;
  }

  function openWine(wine, { searchId = null, context = "" } = {}) {
    if (!wine) return;
    wineDetail.value = { wine, searchId, context };
  }

  async function deleteNotification(id) {
    await apiFetch(`/api/v1/notifications/${id}`, { method: "DELETE" });
    await loadNotifications();
  }

  async function deleteAllNotifications() {
    await apiFetch("/api/v1/notifications", { method: "DELETE" });
    await loadNotifications();
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

  async function markNotificationRead(id) {
    await apiFetch(`/api/v1/notifications/${id}/read`, { method: "POST" });
    await loadNotifications();
  }

  async function openNotificationDetail(id) {
    notificationDetail.value = await apiFetch(`/api/v1/notifications/${id}`);
    await loadNotifications();
  }

  return {
    achievements,
    achievementToasts,
    checkAchievements,
    dismissToast,
    leaderboard,
    loadAchievements,
    activeView,
    addFavorite,
    authDialog,
    pendingAction,
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
    deleteNotification,
    deleteAllNotifications,
    loadReviews,
    markNotificationRead,
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
    reactToReview,
    clearReviewReaction,
    ratingDialog,
    saveReview,
    deleteReview,
    markAllNotificationsRead,
    setSelectedFile,
    showDiagnostics,
    showGenerated,
    similarMatches,
    submitAuth,
    updateProfileSettings,
    uploadAvatar,
    user,
  };
}
