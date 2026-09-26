const apiTarget = process.env.VITE_API_TARGET || process.env.NUXT_API_TARGET || "http://localhost:8000";

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
  devServer: {
    host: "0.0.0.0",
    port: 5173,
  },
});
