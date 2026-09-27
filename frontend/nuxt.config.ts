const apiTarget = process.env.VITE_API_TARGET || process.env.NUXT_API_TARGET || "http://localhost:8000";
// домены/IP, с которых можно открыть dev-сервер (кроме localhost), через запятую; "*" — любые
const allowedHostsEnv = (process.env.VITE_ALLOWED_HOSTS || "").trim();
const allowedHosts =
  allowedHostsEnv === "*" ? true : allowedHostsEnv.split(",").map((host) => host.trim()).filter(Boolean);

export default defineNuxtConfig({
  ssr: false,
  compatibilityDate: "2026-09-26",
  devtools: { enabled: false },
  experimental: {
    appManifest: false,
  },
  css: ["~/assets/css/main.css"],
  modules: ["@nuxtjs/google-fonts"],
  googleFonts: {
    families: {
      "Playfair Display": [400, 500, 600],
    },
    display: "swap",
  },
  app: {
    head: {
      title: "Сканер вин",
      meta: [
        {
          name: "description",
          content: "Сканер российских вин: распознавание этикетки, карточка вина, похожие варианты и история.",
        },
        { name: "viewport", content: "width=device-width, initial-scale=1, viewport-fit=cover" },
      ],
    },
  },
  nitro: {
    routeRules: {
      "/api/**": { proxy: `${apiTarget}/api/**` },
      "/health": { proxy: `${apiTarget}/health` },
    },
  },
  vite: {
    server: { allowedHosts },
  },
  devServer: {
    host: "0.0.0.0",
    port: 5173,
  },
});
