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
  const selectedMatch = useState("wine.selectedMatch", () => null);
  const selectedPhotos = useState("wine.selectedPhotos", () => ({}));
  const showGenerated = useState("wine.showGenerated", () => false);
  const showDiagnostics = useState("wine.showDiagnostics", () => false);

  const history = useState("wine.history", () => ({ items: [] }));
  const favorites = useState("wine.favorites", () => ({ items: [] }));
  const reviews = useState("wine.reviews", () => ({ items: [] }));
  const notifications = useState("wine.notifications", () => ({ items: [], unread: 0 }));
  const notificationDetail = useState("wine.notificationDetail", () => null);
  const initialized = useState("wine.initialized", () => false);

  const bestMatch = computed(() => selectedMatch.value || searchResponse.value?.results?.[0] || null);
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
    if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
    previewUrl.value = URL.createObjectURL(file);
  }

  function handleDrop(event) {
    isDragging.value = false;
    setSelectedFile(event.dataTransfer.files?.[0]);
  }

  async function searchWine() {
    if (!selectedFile.value) return;
    isSearching.value = true;
    errorMessage.value = "";
    selectedMatch.value = null;
    showGenerated.value = false;
    try {
      const form = new FormData();
      form.append("image", selectedFile.value);
      searchResponse.value = await apiFetch("/api/v1/search/image/extended?limit=12", {
        method: "POST",
        body: form,
      });
      if (!searchResponse.value?.results?.length) {
        errorMessage.value = "Вино не найдено. Попробуйте фото этикетки крупнее и без бликов.";
      }
      await loadPublicHistory();
    } catch (error) {
      errorMessage.value = error?.data?.detail || "Не удалось выполнить поиск";
    } finally {
      isSearching.value = false;
    }
  }

  function selectMatch(match) {
    selectedMatch.value = match;
    window.scrollTo({ top: 0, behavior: "smooth" });
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
  }

  async function removeFavorite(wineId) {
    await apiFetch(`/api/v1/favorites/${wineId}`, { method: "DELETE" });
    await loadFavorites();
  }

  async function setReview(match, rating, notificationId = null) {
    if (!user.value) {
      authDialog.value = true;
      return;
    }
    await apiFetch(`/api/v1/reviews/${match.wine_id || match.wine?.id}`, {
      method: "PUT",
      body: { rating, notification_id: notificationId },
    });
    await loadReviews();
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
    selectedFile,
    selectedPhotos,
    selectMatch,
    setReview,
    setSelectedFile,
    showDiagnostics,
    showGenerated,
    similarMatches,
    submitAuth,
    user,
  };
}
