// Разделы и окна приложения в истории браузера: «Назад» ходит по сайту, а не уводит с него.
// Роутера нет (одна страница), поэтому синхронизируем адрес вручную через History API.
const PROFILE_TABS = ["history", "favorites", "reviews", "notifications", "achievements"];
// окна, которые закрываются кнопкой «Назад»; порядок — снизу вверх (карточка вина открывается поверх напоминания)
const MODALS = ["notification", "wine"];

export function useUrlSync() {
  const app = useWineApp();
  let applyingUrl = false;

  function pathFor(view, tab) {
    if (view === "sommelier") return "/sommelier";
    if (view === "profile") return `/profile/${tab}`;
    return "/";
  }

  function applyPath(pathname) {
    const [section, tab] = pathname.split("/").filter(Boolean);
    applyingUrl = true;
    if (section === "sommelier") app.activeView.value = "sommelier";
    else if (section === "profile") {
      app.activeView.value = "profile";
      if (PROFILE_TABS.includes(tab)) app.profileTab.value = tab;
    } else app.activeView.value = "scanner";
    nextTick(() => (applyingUrl = false));
  }

  const modalRefs = {
    notification: app.notificationDetail,
    wine: app.wineDetail,
  };

  function openModals() {
    return MODALS.filter((name) => modalRefs[name].value);
  }

  function onPopState(event) {
    // сначала закрываем окна, которых нет в восстановленной записи истории
    const keep = event.state?.modals || [];
    applyingUrl = true;
    for (const name of MODALS) {
      if (modalRefs[name].value && !keep.includes(name)) modalRefs[name].value = null;
    }
    nextTick(() => (applyingUrl = false));
    applyPath(window.location.pathname);
  }

  function start() {
    // первая загрузка: адрес → состояние (прямая ссылка на /profile/history и т.п.)
    applyPath(window.location.pathname);
    window.history.replaceState({ modals: [] }, "", pathFor(app.activeView.value, app.profileTab.value));
    window.addEventListener("popstate", onPopState);

    watch([app.activeView, app.profileTab], ([view, tab]) => {
      if (applyingUrl) return;
      const path = pathFor(view, tab);
      if (path !== window.location.pathname) window.history.pushState({ modals: [] }, "", path);
    });

    for (const name of MODALS) {
      watch(
        () => Boolean(modalRefs[name].value),
        (isOpen) => {
          if (applyingUrl) return;
          const inHistory = window.history.state?.modals || [];
          if (isOpen && !inHistory.includes(name)) {
            window.history.pushState({ modals: openModals() }, "", window.location.pathname);
          } else if (!isOpen && inHistory.at(-1) === name) {
            // окно закрыли крестиком — убираем его шаг из истории, чтобы «Назад» не открывал пустоту
            window.history.back();
          }
        },
      );
    }
  }

  function stop() {
    window.removeEventListener("popstate", onPopState);
  }

  return { start, stop };
}
